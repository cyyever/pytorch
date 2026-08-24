import click

import torch


@click.command()
@click.option(
    "--input-path",
    type=str,
    default="",
    required=True,
    help="path to the ExportedProgram",
)
@click.option(
    "--output-path",
    type=str,
    default="",
    required=True,
)
def main(
    input_path: str = "",
    output_path: str = "",
) -> None:
    data = {}
    ep = torch.export.load(input_path)
    with torch.no_grad():
        example_inputs = ep.example_inputs[0]
        # Get aot compiled module.
        so_path = torch._inductor.aot_compile(ep.module(), example_inputs)
        runner = torch.fx.Interpreter(ep.module())
        output = runner.run(example_inputs)
        if isinstance(output, (list, tuple)):
            output = list(output)
        else:
            output = [output]

        data.update(
            {
                "model_so_path": so_path,
                "inputs": list(example_inputs),
                "outputs": output,
            }
        )

    torch.save(data, output_path)


if __name__ == "__main__":
    main()
