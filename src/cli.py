import asyncio

import click
from dotenv import load_dotenv

from data.augmentation.main import main as run_augmentation

load_dotenv()


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to SGD dataset root directory",
)
@click.option(
    "--output_dir",
    type=str,
    required=True,
    help="Output directory for results",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of dialogues to process",
)
@click.option(
    "--no_llm",
    is_flag=True,
    help="Disable LLM-based injections (COR, RST)",
)
@click.option(
    "--split",
    type=click.Choice(["train", "dev", "test"]),
    default="train",
    help="Dataset split to process",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="LLM model name for LLM-based injections",
)
@click.option(
    "--provider",
    type=str,
    default="openrouter",
    help="LLM provider name",
)
@click.option(
    "--base_url",
    type=str,
    default="http://localhost:8000/v1",
    help="LLM base URL",
)
@click.option(
    "--tracing",
    is_flag=True,
    help="Enable LLM usage accounting (OpenRouter usage.include)",
)
def run_cli(
    data_dir,
    output_dir,
    limit,
    no_llm,
    split,
    model,
    provider,
    base_url,
    tracing,
):
    """SGD Disfluency Augmentation Pipeline"""
    asyncio.run(
        run_augmentation(
            data_dir=data_dir,
            output_dir=output_dir,
            limit=limit,
            use_llm=not no_llm,
            split=split,
            model=model,
            provider=provider,
            base_url=base_url,
            tracing=tracing,
        )
    )


if __name__ == "__main__":
    run_cli()
