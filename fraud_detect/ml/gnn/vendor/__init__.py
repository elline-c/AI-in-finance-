"""Vendored, minimal Multi-GNN model code (OFFLINE-ONLY).

This package contains model classes and adaptations adapted from
`IBM/Multi-GNN <https://github.com/IBM/Multi-GNN>`_ (Apache-2.0, pinned commit
``252b0252afca109d1d216c411c59ff70753b25fc``). See ``LICENSE`` and
``PROVENANCE.md`` in this directory for attribution and the list of
modifications made under Apache-2.0 §4(b).

Modules
-------
    * :mod:`.models`      — GINe / GATe / PNA / RGCN definitions.
    * :mod:`.adaptations` — reverse message passing, ego IDs, and port
      numberings (``AddEgoIds``, ``add_arange_ids``, ``ports``,
      ``GraphData``, ``HeteroGraphData``, ``create_hetero_obj`` …).

Heavy-dependency boundary
-------------------------
Both submodules depend on ``torch`` / ``torch_geometric``, which are
OFFLINE-ONLY dependencies (``requirements-gnn.txt``). This ``__init__`` module
deliberately performs NO imports of those submodules, so importing the
``vendor`` package itself never pulls in torch. Import the submodules
explicitly from the offline trainer only, e.g.::

    from fraud_detect.ml.gnn.vendor.models import GINe, GATe, PNA, RGCN
    from fraud_detect.ml.gnn.vendor.adaptations import AddEgoIds, create_hetero_obj

Because the live screening path never imports these submodules, torch is never
required at request time.
"""

__all__ = ["models", "adaptations"]
