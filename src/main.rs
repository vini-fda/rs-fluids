#[macro_use]
extern crate nalgebra as na;
use ::rand::{prelude::ThreadRng, thread_rng};
use na::{OMatrix, SMatrix, Scalar};
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use rand_distr::{
    num_traits::{Float, Zero},
    Distribution, Normal, NormalError, StandardNormal,
};
use std::{
    cmp::min,
    ops::{Index, IndexMut, Rem},
    time::Duration,
    rc::Rc
};

use macroquad::prelude::*;

fn draw_square(x: f32, y: f32, l: f32, color: Color) {
    macroquad::shapes::draw_rectangle(x, y, l, l, color);
}

//Double-buffered Static Matrix
struct BufSMatrix<T: Scalar + Float + Zero, const N: usize> {
    current_buffer: BufferIndex,
    pub data0: SMatrix<T, N, N>,
    pub data1: SMatrix<T, N, N>,
}

impl<T: Scalar + Float + Zero, const N: usize> BufSMatrix<T, N> {
    pub fn zeros() -> BufSMatrix<T, N> {
        BufSMatrix::<T, N> {
            current_buffer: BufferIndex::ZERO,
            data0: SMatrix::zeros(),
            data1: SMatrix::zeros(),
        }
    }

    pub fn get_buffer(&self, b: BufferIndex) -> &SMatrix<T, N, N> {
        match b {
            BufferIndex::ZERO => &self.data0,
            BufferIndex::ONE => &self.data1,
        }
    }

    pub fn get_mut_buffer(&mut self, b: BufferIndex) -> &mut SMatrix<T, N, N> {
        match b {
            BufferIndex::ZERO => &mut self.data0,
            BufferIndex::ONE => &mut self.data1,
        }
    }

    pub fn get_current_buffer(&self) -> &SMatrix<T, N, N> {
        self.get_buffer(self.current_buffer)
    }

    pub fn get_mut_current_buffer(&mut self) -> &mut SMatrix<T, N, N> {
        self.get_mut_buffer(self.current_buffer)
    }

    pub fn current_buffer_clone(&mut self) -> SMatrix<T, N, N> {
        match self.current_buffer {
            BufferIndex::ZERO => self.data0.clone(),
            BufferIndex::ONE => self.data1.clone(),
        }
    }

    pub fn swap_buffers(&mut self) {
        self.current_buffer = BufferIndex::other(self.current_buffer);
    }
}

impl<'a, T: Scalar + Float + Zero, const N: usize> Index<(usize, usize)> for BufSMatrix<T, N> {
    type Output = T;

    fn index(&self, pos: (usize, usize)) -> &Self::Output {
        let (i, j) = pos;
        let b = self.current_buffer;
        self.index((i, j, b))
    }
}

impl<'a, T: Scalar + Float + Zero, const N: usize> IndexMut<(usize, usize)> for BufSMatrix<T, N> {
    fn index_mut(&mut self, pos: (usize, usize)) -> &mut Self::Output {
        let (i, j) = pos;
        let b = self.current_buffer;
        self.index_mut((i, j, b))
    }
}

impl<'a, T: Scalar + Float + Zero, const N: usize> Index<(usize, usize, BufferIndex)>
    for BufSMatrix<T, N>
{
    type Output = T;

    fn index(&self, pos: (usize, usize, BufferIndex)) -> &Self::Output {
        let (i, j, b) = pos;
        match b {
            BufferIndex::ZERO => self.data0.index((i, j)),
            BufferIndex::ONE => self.data1.index((i, j)),
        }
    }
}

impl<'a, T: Scalar + Float + Zero, const N: usize> IndexMut<(usize, usize, BufferIndex)>
    for BufSMatrix<T, N>
{
    fn index_mut(&mut self, pos: (usize, usize, BufferIndex)) -> &mut Self::Output {
        let (i, j, b) = pos;
        match b {
            BufferIndex::ZERO => self.data0.index_mut((i, j)),
            BufferIndex::ONE => self.data1.index_mut((i, j)),
        }
    }
}

struct Fluid<const N: usize> {
    data_vx: BufSMatrix<f64, N>,
    data_vy: BufSMatrix<f64, N>,
    data_d: BufSMatrix<f64, N>,
    l: f64,
}

impl<const N: usize> Fluid<N> {
    pub fn new_empty(l: f64) -> Fluid<N> {
        Fluid::<N> {
            data_vx: BufSMatrix::zeros(),
            data_vy: BufSMatrix::zeros(),
            data_d: BufSMatrix::zeros(),
            l,
        }
    }
    pub fn new_random(l: f64, normal: Normal<f64>) -> Fluid<N> {
        let mut rng: ThreadRng = thread_rng();
        //let normal: Normal<T> = Normal::new(T::from(1.0).unwrap(), T::from(0.25).unwrap()).unwrap();
        let iter = normal.sample_iter(&mut rng);
        let mut data_d = BufSMatrix::<f64, N>::zeros();
        data_d.data0 = SMatrix::from_iterator(iter);

        let mut data_vx = BufSMatrix::<f64, N>::zeros();
        // let normal2 = Normal::new(1.0, 5.0).unwrap();
        // let iter = normal2.sample_iter(&mut rng);
        data_vx.data0 = SMatrix::from_fn(|i, j| {
            let i = i as f64 - 64.0;
            let j = j as f64 - 64.0;
            -40.0 * j / (i*i + j*j + 1.0).sqrt()
        });

        let mut data_vy = BufSMatrix::<f64, N>::zeros();
        // let normal3 = Normal::new(0.0, 0.9).unwrap();
        // let iter = normal3.sample_iter(&mut rng);
        data_vy.data0 = SMatrix::from_fn(|i, j| {
            let i = i as f64 - 64.0;
            let j = j as f64 - 64.0;
            40.0 * i / (i*i + j*j + 1.0).sqrt()
        });

        Fluid::<N> {
            data_vx: BufSMatrix::zeros(),
            data_vy,
            data_d,
            l,
        }
    }

    fn swap_buffers(&mut self, dk: DataKind) {
        match dk {
            DataKind::VELOCITY_X => {
                self.data_vx.swap_buffers();
            }
            DataKind::VELOCITY_Y => {
                self.data_vy.swap_buffers();
            }
            DataKind::DENSITY => {
                self.data_d.swap_buffers();
            }
        }
    }

    #[inline]
    fn linear_solve(x: &mut SMatrix<f64, N, N>, b: &SMatrix<f64, N, N>, a: f64, c: f64, m: i32) {
        let max_iters = 20;
        for _ in 1..max_iters {
            for i in 1..(N - 1) {
                for j in 1..(N - 1) {
                    x[(i, j)] = (b[(i, j)]
                        + a * (x[(i - 1, j)] + x[(i + 1, j)] + x[(i, j - 1)] + x[(i, j + 1)]))
                        / c;
                }
            }
            Self::set_bnd(m, x);
        }
    }

    fn density_step(&mut self, dt: f64) {
        let k = 3.0;
        let f = k * dt / (self.l.powf(2.0));
        let d0 = self.data_d.current_buffer_clone();
        self.data_d.swap_buffers();{
        let d = self[DataKind::DENSITY].get_mut_current_buffer();
        Self::linear_solve(d, &d0, f, 1.0 + 4.0 * f, 0);}

        // Advect density
        self.advect(DataKind::DENSITY, dt);
    }

    fn velocity_step(&mut self, dt: f64) {
        let k = 0.01;
        let f = k * dt / (self.l.powf(2.0));
        // Diffusion on velocities
        let vx0 = self.data_vx.current_buffer_clone();
        let vy0 = self.data_vy.current_buffer_clone();
        self.data_vx.swap_buffers();
        self.data_vy.swap_buffers();
        {
            let vx = self.data_vx.get_mut_current_buffer();
            Self::linear_solve(vx, &vx0, f, 1.0 + 4.0 * f, 1);
            let vy = self.data_vy.get_mut_current_buffer();
            Self::linear_solve(vy, &vy0, f, 1.0 + 4.0 * f, 2);
        }

        // Advect on vel x
        self.advect(DataKind::VELOCITY_X, dt);
        // Advect on vel y
        self.advect(DataKind::VELOCITY_Y, dt);
        // Projection
        {
            let vx = self.data_vx.get_mut_current_buffer();
            let vy = self.data_vy.get_mut_current_buffer();
            Self::project(vx, vy, self.l);
            Self::set_bnd(1, vx);
            Self::set_bnd(2, vy);
        }
    }

    fn project(
        vx: &mut SMatrix<f64, N, N>,
        vy: &mut SMatrix<f64, N, N>,
        h: f64
    ) {
        let mut div = SMatrix::<f64, N, N>::zeros();
        let mut p =  SMatrix::<f64, N, N>::zeros();
        // Compute divergence
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                div[(i, j)] =
                    -0.5 * h * (vx[(i + 1, j)] - vx[(i - 1, j)] + vy[(i, j + 1)] - vy[(i, j - 1)]);
                //p[(i, j)] = 0.0; zero
            }
        }
        // Solve Poisson eq
        Self::linear_solve(&mut p, &div, 1.0, 4.0, 0);
        // Subtract gradient field
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                vx[(i, j)] -= 0.5 * (p[(i + 1, j)] - p[(i - 1, j)]) / h;
                vy[(i, j)] -= 0.5 * (p[(i, j + 1)] - p[(i, j - 1)]) / h;
            }
        }
    }

    fn set_bnd(b: i32, x: &mut SMatrix<f64, N, N>) {
        for i in 1..(N - 1) {
            x[(0, i)] = if b == 1 { -x[(1, i)] } else { x[(1, i)] };

            x[(N - 1, i)] = if b == 1 {
                -x[(N - 2, i)]
            } else {
                x[(N - 2, i)]
            };

            x[(i, 0)] = if b == 2 { -x[(i, 1)] } else { x[(i, 1)] };

            x[(i, N - 1)] = if b == 2 {
                -x[(i, N - 2)]
            } else {
                x[(i, N - 2)]
            };
        }
        x[(0, 0)] = 0.5 * (x[(1, 0)] + x[(0, 1)]);
        x[(0, N - 1)] = 0.5 * (x[(1, N - 1)] + x[(0, N - 2)]);
        x[(N - 1, 0)] = 0.5 * (x[(N - 2, 0)] + x[(N - 1, 1)]);
        x[(N - 1, N - 1)] = 0.5 * (x[(N - 2, N - 1)] + x[(N - 1, N - 2)]);
    }

    fn clamp(x: f64, a: f64, b: f64) -> f64 {
        if x < a {
            a
        } else if x > b {
            b
        } else {
            x
        }
    }

    fn advect(&mut self, dk: DataKind, dt: f64) {
        let x0: SMatrix<f64, N, N>;
        let x1: &mut SMatrix<f64, N, N>;
        let h = self.l;

        let vx = self.data_vx.current_buffer_clone();
        let vy = self.data_vy.current_buffer_clone();
        match dk {
            DataKind::VELOCITY_X => { 
                x0 = self.data_vx.current_buffer_clone();
                self.data_vx.swap_buffers();
                x1 = self[DataKind::VELOCITY_X].get_mut_current_buffer();
            },
            DataKind::VELOCITY_Y => {
                x0 = self.data_vy.current_buffer_clone();
                self.data_vy.swap_buffers();
                x1 = self[DataKind::VELOCITY_Y].get_mut_current_buffer();
            }, 
            DataKind::DENSITY => {
                x0 = self.data_d.current_buffer_clone();
                self.data_d.swap_buffers();
                x1 = self[DataKind::DENSITY].get_mut_current_buffer();
            }
        }

        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                let mut x = (i as f64) - dt * vx[(i,j)] / h;
                let mut y = (j as f64) - dt * vy[(i,j)] / h;
                x = Self::clamp(x, 0.5, N as f64 - 1.5);
                y = Self::clamp(y, 0.5, N as f64 - 1.5);
                let i0 = x.floor();
                let j0 = y.floor();
                let i1 = i0 + 1.0;
                let j1 = j0 + 1.0;
                let (s0, t0) = (x - i0, y - j0);
                let (s1, t1) = (1.0 - s0, 1.0 - t0);

                let (i0, j0) = (i0 as usize, j0 as usize);
                let (i1, j1) = (i1 as usize, j1 as usize);

                x1[(i, j)] = t1 * (s1 * x0[(i0, j0)] + s0 * x0[(i1, j0)])
                    + t0 * (s1 * x0[(i0, j1)] + s0 * x0[(i1, j1)]);
            }
        }
    }

    pub fn draw(&self) {
        let l = self.l as f32;
        for i in 0..N {
            for j in 0..N {
                let c = self[(i, j, DataKind::DENSITY)];
                let mut c = (c * 100.0).floor() as i64;
                if c < 0 {
                    c = 0;
                } else {
                    c = min(c, 255);
                }
                let c = c as u8;
                let i = i as f32;
                let j = j as f32;
                draw_square(
                    l * i + 50.0,
                    l * j + 50.0,
                    l,
                    macroquad::color_u8!(c, c, c, 255),
                )
            }
        }
    }
}

impl<const N: usize> Index<(usize, usize, DataKind)> for Fluid<N> {
    type Output = f64;

    fn index(&self, pos: (usize, usize, DataKind)) -> &Self::Output {
        let (i, j, dk) = pos;
        match dk {
            DataKind::VELOCITY_X => &self.data_vx[(i, j)],
            DataKind::VELOCITY_Y => &self.data_vy[(i, j)],
            DataKind::DENSITY => &self.data_d[(i, j)],
        }
    }
}

impl<const N: usize> IndexMut<(usize, usize, DataKind)> for Fluid<N> {
    fn index_mut(&mut self, pos: (usize, usize, DataKind)) -> &mut Self::Output {
        let (i, j, dk) = pos;
        match dk {
            DataKind::VELOCITY_X => &mut self.data_vx[(i, j)],
            DataKind::VELOCITY_Y => &mut self.data_vy[(i, j)],
            DataKind::DENSITY => &mut self.data_d[(i, j)],
        }
    }
}

impl<const N: usize> Index<DataKind> for Fluid<N> {
    type Output = BufSMatrix<f64, N>;

    fn index(&self, index: DataKind) -> &Self::Output {
        match index {
            DataKind::VELOCITY_X => &self.data_vx,
            DataKind::VELOCITY_Y => &self.data_vy,
            DataKind::DENSITY => &self.data_d,
        }
    }
}

impl<const N: usize> IndexMut<DataKind> for Fluid<N> {
    fn index_mut(&mut self, index: DataKind) -> &mut Self::Output {
        match index {
            DataKind::VELOCITY_X => &mut self.data_vx,
            DataKind::VELOCITY_Y => &mut self.data_vy,
            DataKind::DENSITY => &mut self.data_d,
        }
    }
}

#[derive(Clone, Copy)]
enum BufferIndex {
    ZERO,
    ONE,
}

impl BufferIndex {
    pub fn other(b: BufferIndex) -> BufferIndex {
        match b {
            BufferIndex::ZERO => BufferIndex::ONE,
            BufferIndex::ONE => BufferIndex::ZERO,
        }
    }
}

enum DataKind {
    VELOCITY_X,
    VELOCITY_Y,
    DENSITY,
}

#[macroquad::main("egui with macroquad")]
async fn main() {
    let normal = Normal::new(1.0, 0.5).unwrap();
    let mut my_fluid: Fluid<120> = Fluid::new_random(5.0, normal);
    let mut timestep: u64 = 0;
    let mut t = 0.0;
    let mut new_t = 1.0;
    let mut δt = 1.0;
    let mut wait_t = 1.0;
    loop {
        t = get_time();
        clear_background(WHITE);

        // Process keys, mouse etc.

        egui_macroquad::ui(|egui_ctx| {
            egui::Window::new("rs ❤ fluids").show(egui_ctx, |ui| {
                ui.label("Test");
                //if ui.button("Advance timestep").clicked() {
                if wait_t > 0.050 {
                    timestep += 1;
                    my_fluid.density_step(wait_t);
                    my_fluid.velocity_step(wait_t);
                    wait_t = 0.0;
                }

                //}
                ui.label(format!("Timestep: {}", timestep));
            });
        });

        // Draw things before egui
        my_fluid.draw();

        egui_macroquad::draw();

        // Draw things after egui

        next_frame().await;
        new_t = get_time();
        δt = new_t - t;
        wait_t += δt;
    }
}

fn OneDiagMatrix<const N: usize>(di: i32, dj: i32) -> SMatrix<f64, N, N> {
    SMatrix::<f64, N, N>::from_fn(|i, j| {
        let i = i as i32;
        let j = j as i32;
        ((i + di) == (j + dj)) as u8 as f64
    })
}

// fn main() {
//     let mut rng: ThreadRng = thread_rng();
//     let normal = Normal::new(2.0, 0.5).unwrap();
//     let iter = normal.sample_iter(&mut rng);
//     let M = OneDiagMatrix::<10>(0, 1) + OneDiagMatrix::<10>(1, 0);
//     let A = SMatrix::<f64, 10, 10>::from_iterator(iter);
//     println!("{:}", M);
//     println!("{:.2}", A);
//     println!("{:.2}", M*A);
//     println!("{:.2}", A*M);
// }