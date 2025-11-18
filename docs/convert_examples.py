"""Convert Python example files to markdown for MkDocs using jupytext and nbconvert.

1. Convert .py files to markdown using jupytext -> nbconvert pipeline
2. Execute notebooks during conversion for live output
3. Run conversion in the examples/ directory so imports work correctly
4. Write generated markdown and images to docs/_examples/ via mkdocs_gen_files
5. Clean up temporary files from examples/ directory after conversion
"""

import shutil
import subprocess
from pathlib import Path

from loguru import logger
from mkdocs_gen_files import open as gen_open

# Files to exclude from conversion (supporting files)
EXCLUDE = [
    "helper.py",
    "plotting.py",
    "ex_helper.py",
    "ex_regression.py",
    "ex_classification.py",
]


def _collect_image_files(examples_dir: Path, stem: str) -> list[Path]:
    """Collect generated image files from nbconvert output.

    Args:
      examples_dir: Directory where example files are located.
      stem: Stem of the original Python file.

    Returns:
      List of image file paths.
    """
    files_dir = examples_dir / f"{stem}_files"

    if not files_dir.exists():
        return []

    image_files = [
        img_file
        for img_file in files_dir.iterdir()
        if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif"}
    ]

    if image_files:
        logger.info(f"Found {len(image_files)} image files in {files_dir.name}/")

    return image_files


def _cleanup_generated_files(examples_dir: Path, stem: str) -> None:
    """Remove generated markdown and image files from examples directory.

    Args:
      examples_dir: Directory where example files are located.
      stem: Stem of the original Python file.
    """
    md_file = examples_dir / f"{stem}.md"
    if md_file.exists():
        md_file.unlink()

    files_dir = examples_dir / f"{stem}_files"
    if files_dir.exists():
        shutil.rmtree(files_dir)


def process_file(file: Path, examples_dir: Path) -> tuple[str | None, list[Path]]:
    """Convert a python file to markdown using jupytext and nbconvert.

    Args:
      file: Python file to convert.
      examples_dir: Directory where example files are located.

    Returns:
      Tuple of markdown content (or None if failed) and list of image files.
    """
    md_file = examples_dir / f"{file.stem}.md"

    # Pipeline: jupytext -> nbconvert with execution
    # Run in examples directory so imports work
    jupytext_cmd = [
        "uv",
        "run",
        "jupytext",
        "--to",
        "ipynb",
        file.name,
        "--output",
        "-",
    ]
    nbconvert_cmd = [
        "uv",
        "run",
        "jupyter",
        "nbconvert",
        "--to",
        "markdown",
        "--execute",
        "--stdin",
        "--output",
        md_file.name,
    ]

    # Run jupytext first
    jupytext_result = subprocess.run(  # noqa: S603
        jupytext_cmd, capture_output=True, text=True, cwd=examples_dir, check=False
    )

    if jupytext_result.returncode != 0:
        logger.warning(f"Failed to convert {file.name}")
        logger.warning(f"Error: {jupytext_result.stderr}")
        return None, []

    # Pipe output to nbconvert
    result = subprocess.run(  # noqa: S603
        nbconvert_cmd,
        input=jupytext_result.stdout,
        capture_output=True,
        text=True,
        cwd=examples_dir,
        check=False,
    )

    if result.returncode != 0:
        logger.warning(f"Failed to convert {file.name}")
        logger.warning(f"Error: {result.stderr}")
        return None, []

    if not md_file.exists():
        logger.warning(f"Output file {md_file} was not created")
        return None, []

    content = Path.read_text(md_file)
    image_files = _collect_image_files(examples_dir, file.stem)
    logger.info(f"✓ Converted {file.name} -> markdown content")
    return content, image_files


def _write_markdown_and_images(
    py_file: Path, markdown_content: str, image_files: list
) -> None:
    """Write markdown file and associated images using MkDocs gen-files API.

    Args:
      py_file: Original Python file.
      markdown_content: Converted markdown content.
      image_files: List of image files to copy.
    """
    md_path = f"_examples/{py_file.stem}.md"
    with gen_open(md_path, "w") as f:
        f.write(markdown_content)
    logger.info(f"✓ Generated {md_path}")

    for img_file in image_files:
        img_rel_path = f"_examples/{py_file.stem}_files/{img_file.name}"
        data = img_file.read_bytes()
        with gen_open(img_rel_path, "wb") as f_out:
            f_out.write(data)
        logger.info(f"✓ Copied image {img_rel_path}")


def main():
    """Main function to convert Python examples to markdown."""
    repo_root = Path(__file__).parent.parent
    examples_dir = repo_root / "examples"

    example_files = [f for f in examples_dir.glob("*.py") if f.name not in EXCLUDE]
    logger.info(f"Found {len(example_files)} example files to convert")
    logger.info(f"Excluded files: {', '.join(sorted(EXCLUDE))}")

    success_count = 0
    for py_file in example_files:
        logger.info(f"Processing {py_file.name}...")

        try:
            markdown_content, image_files = process_file(py_file, examples_dir)

            if markdown_content:
                _write_markdown_and_images(py_file, markdown_content, image_files)
                success_count += 1
        finally:
            # Clean up generated files from examples directory
            _cleanup_generated_files(examples_dir, py_file.stem)

    logger.info(
        f"Conversion complete! {success_count}/{len(example_files)} files converted successfully"  # noqa: E501
    )


main()
