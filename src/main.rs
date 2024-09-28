use std::cmp::max;
use std::path::Path;
use rand::{random, Rng};

const INPUT_SIZE: u32 = 784;
const HIDDEN_SIZE: u32 = 256;
const OUTPUT_SIZE: u32 = 10;
const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u16 = 20;
const BATCH_SIZE: u16 = 64;
const IMAGE_SIZE: u16 = 28;
const TRAIN_SPLIT: f32 = 0.8;

const TRAIN_IMG_PATH: &str = "data/train-images.idx3-ubyte";
const TRAIN_LBL_PATH: &str = "data/train-labels.idx1-ubyte";

struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: u32,
    output_size: u32,
}

struct Network {
    hidden: Layer,
    output: Layer,
}

struct InputData {
    images: Vec<u8>,
    labels: Vec<u8>,
    num_images: u32,
}

impl Layer {
    pub fn new(in_size: u32, out_size: u32) -> Self {
        let n = in_size * out_size;
        let scale = (2.0f32 / in_size as f32).sqrt();

        let mut weights = vec![0f32; n as usize];

        for weight in weights.iter_mut() {
            *weight = (random::<f32>() / f32::MAX - 0.5) * 2.0 * scale;
        }

        Layer {
            weights,
            biases: vec![0.0; out_size as usize],
            input_size: in_size,
            output_size: out_size,
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for (i, output_value) in output.iter_mut().enumerate() {
            *output_value = self.biases[i];

            for j in 0..self.input_size as usize {
                *output_value += input[j] * self.weights[j * self.output_size as usize + i];
            }
        }
    }

    pub fn backward(&mut self, input: &[f32], output_grad: &[f32], input_grad: &mut [f32], lr: f32) {
        for (i, &grad) in output_grad.iter().enumerate() {
            for (j, &input_val) in input.iter().enumerate() {
                let idx = j * self.output_size as usize + i;
                let grad = grad * input_val;
                self.weights[idx] -= lr * grad;

                if input_grad.len() > 0 {
                    input_grad[j] += grad * self.weights[idx];
                }

            }
            self.biases[i] -= lr * grad;
        }
    }
}

impl Network {
    pub fn new(input: u32, hidden: u32, output: u32) -> Self {
        Network {
            hidden: Layer::new(input, hidden),
            output: Layer::new(hidden, output),
        }
    }

    pub fn train(&mut self, input: &[f32], label: u16, lr: f32) {
        let mut hidden_output = vec![0.0f32; HIDDEN_SIZE as usize];
        let mut final_output = vec![0.0f32; OUTPUT_SIZE as usize];
        let mut output_grad = vec![0.0f32; OUTPUT_SIZE as usize];
        let mut hidden_grad = vec![0.0f32; HIDDEN_SIZE as usize];

        self.hidden.forward(input, &mut hidden_output);
        for weight in self.hidden.weights.iter_mut() {
            if *weight <= 0f32 { // ReLU
                *weight = 0f32;
            }
        }

        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        for (index, output_grad_value) in output_grad.iter_mut().enumerate() {
            let same: f32 = if index == label as usize {
                0.0
            } else {
                1.0
            };
            *output_grad_value = final_output[index] - same;
        }

        self.output.backward(&hidden_output, &output_grad, &mut hidden_grad, lr);

        for hidden_val in hidden_grad.iter_mut() {
            if *hidden_val <= 0f32 { // ReLU derivative
                *hidden_val = 0.0;
            }
        }

        self.hidden.backward(&input, &hidden_grad, &mut vec![], lr);
    }

    pub fn predict(&self, input: &[f32]) -> u16 {
        let mut hidden_output = vec![0.0f32; HIDDEN_SIZE as usize];
        let mut final_output = vec![0.0f32; OUTPUT_SIZE as usize];

        self.hidden.forward(input, &mut hidden_output);

        for weight in hidden_output.iter_mut() {
            if *weight <= 0f32 { // ReLU
                *weight = 0f32;
            }
        }

        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        let mut max_index = 0;
        for (index, output_value) in final_output.iter().enumerate() {
            if *output_value > final_output[max_index] {
                max_index = index;
            }
        }

        max_index as u16
    }
}

#[inline]
fn bytes_to_u32(data: &[u8]) -> u32 {
    let array: [u8; 4] = data.try_into().unwrap();
    u32::from_be_bytes(array)
}

impl InputData {
    pub fn load(data: &Path, labels: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data_contents = std::fs::read(data)?;
        let labels_contents = std::fs::read(labels)?;

        let num_images = bytes_to_u32(&data_contents[4..8]);
        let num_rows = bytes_to_u32(&data_contents[8..12]);
        let num_cols = bytes_to_u32(&data_contents[12..16]);

        let num_labels = bytes_to_u32(&labels_contents[4..8]);

        #[cfg(debug_assertions)]
        println!("Num images: {num_images}, Num rows: {num_rows}, Num cols: {num_cols}, Num labels: {num_labels}");

        debug_assert_eq!(num_images, num_labels);

        Ok(Self {
            images: data_contents[16..].to_vec(),
            labels: labels_contents[8..].to_vec(),
            num_images,
        })
    }

    pub fn shuffle(&mut self) {
        for i in self.num_images as usize - 1..0 {
            let j = random::<usize>() % (i + 1);
            for k in 0..INPUT_SIZE as usize {
                self.images.swap(i * INPUT_SIZE as usize + k, j * INPUT_SIZE as usize + k);
            }
            self.labels.swap(i, j);
        }
    }
}

fn softmax(input: &mut[f32]) {
    let mut max = input[0];
    let mut sum = 0f32;

    for input_value in input.iter() {
        if *input_value > max {
            max = *input_value;
        }
        sum += *input_value;
    }

    for input_value in input.iter_mut() {
        *input_value = (*input_value - max).exp();
        *input_value /= sum;
    }
}

fn main() {
    let mut net = Network::new(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    let mut data = InputData::load(TRAIN_IMG_PATH.as_ref(), TRAIN_LBL_PATH.as_ref()).expect("Failed to load mnist data");
    data.shuffle();

    let train_size = (data.num_images as f32 * TRAIN_SPLIT) as usize;
    let test_size = data.num_images as usize - train_size;
    let mut img = vec![0.0f32; INPUT_SIZE as usize];

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;

        for i in (0..train_size).step_by(BATCH_SIZE as usize) {
            let mut j = 0;

            let idx = i + j;
            for (k, img_val) in img.iter_mut().enumerate() { // Data normalization
                *img_val = data.images[idx * INPUT_SIZE as usize + k] as f32 / 255.0f32;
            }

            net.train(&img, data.labels[idx] as u16, LEARNING_RATE);

            let mut hidden_output = vec![0.0f32; HIDDEN_SIZE as usize];
            let mut final_output = vec![0.0f32; OUTPUT_SIZE as usize];
            net.hidden.forward(&img, &mut hidden_output);
            for hidden_output_val in hidden_output.iter_mut() {
                if *hidden_output_val <= 0.0 { // ReLU
                    *hidden_output_val = 0.0;
                }
            }
            net.output.forward(&hidden_output, &mut final_output);
            softmax(&mut final_output);

            total_loss += -(final_output[data.labels[idx] as usize] + 1e-10f32).ln();

            loop {
                if j >= BATCH_SIZE as usize && i + j >= train_size {
                    break;
                }
                j += 1;
            }
        }

        let mut correct = 0;
        for i in train_size..data.num_images as usize {
            for (k, img_val) in img.iter_mut().enumerate() { // Data normalization
                *img_val = data.images[i * INPUT_SIZE as usize + k] as f32 / 255.0f32;
            }
            
            if net.predict(&img) == data.labels[i] as u16 {
                correct += 1;
            }
        }
        println!("Epoch {}, Accuracy: {}, Avg Loss: {}", epoch + 1, correct as f32 / test_size as f32 * 100f32, total_loss / train_size as f32);
    }
}