"""OTX CLI ArgParser."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, TypeVar

import docstring_parser
import yaml
from jsonargparse import (
    ArgumentParser,
    Namespace,
    class_from_function,
)
from jsonargparse._loaders_dumpers import DefaultLoader

from .help_formatter import OTXHelpFormatter


def tuple_constructor(loader: DefaultLoader, node: yaml.SequenceNode) -> tuple | None:
    """Construct a tuple from a YAML sequence node.

    Args:
        loader (DefaultLoader): The YAML loader.
        node (yaml.SequenceNode): The YAML sequence node.

    Returns:
        Optional[tuple]: The constructed tuple, or None if the node is not a sequence.
    """
    if isinstance(node, yaml.SequenceNode):
        # Load the elements as a list
        elements = loader.construct_sequence(node)
        # Return the tuple
        return tuple(elements)
    return None


DefaultLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


def get_short_docstring(component: TypeVar) -> str | None:
    """Get the short description from the docstring.

    Args:
        component (object): The component to get the docstring from

    Returns:
        Optional[str]: The short description
    """
    if component.__doc__ is None:
        return None
    docstring = docstring_parser.parse(component.__doc__)
    return docstring.short_description


class OTXArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for OTX."""

    def __init__(
        self,
        *args,
        description: str = "OpenVINO Training-Extension command line tool",
        env_prefix: str = "otx",
        default_env: bool = False,
        default_config_files: list[str | None] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the ArgumentParser object for the OpenVINO Training-Extension command line tool.

        Args:
            *args: Variable length argument list.
            description (str): Description of the command line tool.
            env_prefix (str): Prefix for environment variables.
            default_env (bool): Whether to use default environment variables.
            default_config_files (Optional[List[Optional[str]]]): List of default configuration files.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(
            *args,
            description=description,
            env_prefix=env_prefix,
            default_env=default_env,
            default_config_files=default_config_files,
            formatter_class=OTXHelpFormatter,
            **kwargs,
        )

    def add_core_class_args(
        self,
        api_class: Callable,
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
        instantiate: bool = False,
        dataclass_mode: bool = False,
    ) -> list[str]:
        """Add arguments from a class to a nested key of the parser.

        Args:
            api_class: A callable or any subclass.
            nested_key (str): Name of the nested namespace to store arguments.
            subclass_mode (bool): Whether allow any subclass of the given class. Default to False.
            required (bool): Whether the argument group is required. Default to True.
            instantiate (bool): Whether api_class for instantiate. Default to False.
            dataclass_mode (bool):  Whether api_class is dataclass_mode. Default to False.

        Returns:
            List[str]: A list with the names of the class arguments added.
        """
        if callable(api_class) and not isinstance(api_class, type):
            api_class = class_from_function(api_class)

        if isinstance(api_class, type):
            if subclass_mode:
                return self.add_subclass_arguments(api_class, nested_key, fail_untyped=False, required=required)
            if dataclass_mode:
                return self.add_dataclass_arguments(
                    api_class,
                    nested_key,
                    fail_untyped=False,
                )
            return self.add_class_arguments(
                api_class,
                nested_key,
                fail_untyped=False,
                instantiate=instantiate,
                sub_configs=True,
            )
        raise NotImplementedError

    def check_config(
        self,
        cfg: Namespace,
        skip_none: bool = True,
        skip_required: bool = True,
        branch: str | None = None,
    ) -> None:
        """Check the configuration for required and optional arguments.

        Args:
            cfg (Namespace): The configuration object to check.
            skip_none (bool, optional): Whether to skip arguments with a value of None. Defaults to True.
            skip_required (bool, optional): Whether to skip required arguments. Defaults to True.
            branch (Optional[str], optional): The branch to check against. Defaults to None.

        Returns:
            None
        """
        # Skip This one for Flexible Configuration