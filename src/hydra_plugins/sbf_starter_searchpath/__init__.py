"""Hydra finds its configs in ``configs/`` (the organisers') and in ``my/configs/`` (yours), from any entry point.

Hydra loads every module of the ``hydra_plugins`` namespace it finds on the path; this one appends both directories to
the config search path of every Hydra app run in this environment. An example (``config_path="../../configs"``) then
takes a config of yours by name (``--config-name=my_search``, with ``defaults: [06_policy_search, _self_]`` in
``my/configs/my_search.yaml``), and a script of yours under ``my/scripts/`` (``config_path="../configs"``) composes
``main`` and the ``task`` group as the examples do. The entry point's own ``config_path`` still comes first.
"""

from pathlib import Path

import rootutils
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


ROOT = Path(rootutils.find_root(search_from=__file__, indicator=".project-root"))


class StarterSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="sbf-starter", path=f"file://{ROOT / 'configs'}")
        search_path.append(provider="sbf-starter-mine", path=f"file://{ROOT / 'my' / 'configs'}")
