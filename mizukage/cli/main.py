"""mizukage CLI — Light L16 LRI/LRIS file reader."""
import click

from mizukage.cli.commands.calib import calib
from mizukage.cli.commands.calib_view import calib_view
from mizukage.cli.commands.info import info
from mizukage.cli.commands.extract import extract
from mizukage.cli.commands.export import export


@click.group()
@click.version_option(package_name="mizukage")
def cli() -> None:
    """mizukage: Light L16 LRI/LRIS camera file reader.

    Read metadata, extract raw Bayer data, and export images from
    Light L16 multi-camera LRI files.
    """


cli.add_command(calib)
cli.add_command(calib_view)
cli.add_command(info)
cli.add_command(extract)
cli.add_command(export)
