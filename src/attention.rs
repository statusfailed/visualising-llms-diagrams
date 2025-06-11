use catgrad::core::nn::layers::*;
use catgrad::core::{Dtype, NdArrayType, Operation, Shape};
use open_hypergraphs::lax::{OpenHypergraph, functor::*, var, var::Var};

use std::cell::RefCell;
use std::rc::Rc;

// 1. Create an OpenHypergraph for an attention layer,
// 2. Turn explicit copy operations into *nodes* in the hypergraph
// 3. Save as an SVG.
pub fn main() -> std::io::Result<()> {
    let arrow = attention_arrow();
    let arrow = var::forget::Forget.map_arrow(&arrow);
    save_svg(&arrow, "images/attention.svg")
}

pub fn attention(
    builder: &Rc<RefCell<OpenHypergraph<NdArrayType, Operation>>>,
    dim: usize,
    name: &str,
    x: Var<NdArrayType, Operation>,
) -> Var<NdArrayType, Operation> {
    let num_heads = 4;
    let head_dim = dim / num_heads;
    let b = x.label.shape.0[0];
    let s = x.label.shape.0[1];

    let k = linear(builder, dim, dim, &format!("{name}.key"), x.clone());
    let q = linear(builder, dim, dim, &format!("{name}.query"), x.clone());
    let v = linear(builder, dim, dim, &format!("{name}.value"), x);

    let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
    let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
    let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

    let q = transpose(builder, 1, 2, q);
    let k = transpose(builder, 1, 2, k);
    let v = transpose(builder, 1, 2, v);

    let tk = transpose(builder, 2, 3, k);
    let attn = mat_mul(builder, q, tk);
    let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
    let attn = attn / denom;
    let attn = softmax(builder, attn);
    let attn = mat_mul(builder, attn, v);
    let x = transpose(builder, 1, 2, attn);
    let x = reshape(builder, Shape(vec![b, s, dim]), x);
    linear(builder, dim, dim, &format!("{name}.proj"), x)
}

// Build the open hypergraph by creating a Var and calling the attention function
fn attention_arrow() -> OpenHypergraph<NdArrayType, Operation> {
    let dim = 8;
    let name = "attention";
    var::build(|state| {
        let x = Var::new(
            state.clone(),
            NdArrayType::new(Shape(vec![1, 1, 8]), Dtype::F32),
        );
        let y = attention(&state, dim, name, x.clone());
        (vec![x], vec![y])
    })
    .unwrap()
}

use graphviz_rust::cmd::{CommandArg, Format};

// Render an OpenHypergraph to an SVG using `open-hypergraphs-dot`
fn save_svg(arrow: &OpenHypergraph<NdArrayType, Operation>, filename: &str) -> std::io::Result<()> {
    let dot_graph = open_hypergraphs_dot::generate_dot(arrow);
    let png_bytes = graphviz_rust::exec(
        dot_graph,
        &mut graphviz_rust::printer::PrinterContext::default(),
        vec![CommandArg::Format(Format::Svg)],
    )?;
    std::fs::write(filename, png_bytes)?;

    Ok(())
}
