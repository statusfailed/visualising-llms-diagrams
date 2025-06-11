use catgrad::core::{NdArrayType, Operation};
use open_hypergraphs::lax::*;

// Save only using tensor shapes
pub fn save_svg<P: AsRef<std::path::Path>>(
    arrow: &OpenHypergraph<NdArrayType, Operation>,
    filename: P,
) -> std::io::Result<()> {
    use graphviz_rust::{
        cmd::{CommandArg, Format},
        exec,
        printer::PrinterContext,
    };

    // Only show shapes on plots
    let opts = open_hypergraphs_dot::Options::<NdArrayType, _> {
        node_label: Box::new(|n| format!("{:?}", n.shape.0)),
        theme: open_hypergraphs_dot::Theme {
            bgcolor: "#1C1C1C".to_string(),
            fontcolor: "#EAEEF4".to_string(),
            color: "#EAEEF4".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    let dot_graph = open_hypergraphs_dot::generate_dot_with(arrow, &opts);

    use graphviz_rust::printer::DotPrinter;
    println!("{}", dot_graph.print(&mut PrinterContext::default()));

    let png_bytes = exec(
        dot_graph,
        &mut PrinterContext::default(),
        vec![CommandArg::Format(Format::Svg)],
    )?;
    std::fs::write(filename, png_bytes)?;

    Ok(())
}
