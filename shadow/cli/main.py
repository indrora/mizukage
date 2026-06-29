"""shadow CLI — Light L16 LRI/LRIS file reader."""
import click

from shadow.cli.commands.calib import calib
from shadow.cli.commands.info import info
from shadow.cli.commands.extract import extract
from shadow.cli.commands.export import export


@click.group()
@click.version_option(package_name="shadow")
def cli() -> None:
    """shadow: Light L16 LRI/LRIS camera file reader.

    Read metadata, extract raw Bayer data, and export images from
    Light L16 multi-camera LRI files.
    """


cli.add_command(calib)
cli.add_command(info)
cli.add_command(extract)
cli.add_command(export)
