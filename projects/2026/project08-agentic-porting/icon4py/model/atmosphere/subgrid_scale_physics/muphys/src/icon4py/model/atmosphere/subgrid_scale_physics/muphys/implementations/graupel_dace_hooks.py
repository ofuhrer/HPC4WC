# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from collections.abc import Sequence
from typing import Any

import dace
from dace import (
    nodes as dace_nodes,
    sdfg as dace_sdfg,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from gt4py.next import config as gtx_config
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def _cleanup_local_self_update(
    *,
    scan_sdfg: dace.SDFG,
    if_stmt_node: dace.sdfg.state.ConditionalBlock,
    if_stmt_conn: str,
    compute_src_node: dace_nodes.AccessNode,
    compute_dst_node: dace_nodes.AccessNode,
    update_src_node: dace_nodes.AccessNode,
    update_dst_node: dace_nodes.AccessNode,
    scan_compute_st: dace.SDFGState,
    scan_update_st: dace.SDFGState,
) -> None:
    temp_data_name = compute_dst_node.data
    assert isinstance(scan_sdfg.arrays[temp_data_name], dace.data.Scalar)
    assert scan_sdfg.arrays[temp_data_name] == compute_src_node.desc(scan_sdfg)
    assert scan_sdfg.arrays[temp_data_name] == update_dst_node.desc(scan_sdfg)

    # reroute the write edge in the compute state
    new_compute_dst_node = scan_compute_st.add_access(compute_src_node.data)
    scan_compute_st.add_edge(
        if_stmt_node,
        if_stmt_conn,
        new_compute_dst_node,
        None,
        dace.Memlet(data=new_compute_dst_node.data, subset="0"),
    )
    for edge in scan_compute_st.out_edges(compute_dst_node):
        scan_compute_st.add_edge(
            new_compute_dst_node,
            None,
            edge.dst,
            edge.dst_conn,
            dace.Memlet(
                data=new_compute_dst_node.data,
                subset=edge.data.get_src_subset(edge, scan_compute_st),
                other_subset=edge.data.get_dst_subset(edge, scan_compute_st),
            ),
        )
    scan_compute_st.remove_node(compute_dst_node)

    # reroute the write edge in the update state
    scan_update_st.add_nedge(
        scan_update_st.add_access(compute_src_node.data),
        update_dst_node,
        dace.Memlet(
            data=compute_src_node.data,
            subset="0",
            other_subset="0",
        ),
    )
    scan_update_st.remove_node(update_src_node)

    # now it is safe to remove the data descriptor
    scan_sdfg.remove_data(temp_data_name, validate=gtx_config.DEBUG)
    print(
        f"Removed self-copy in {if_stmt_node.label}: {compute_src_node.data} -> {compute_dst_node.data}"
    )


def _replace_scan_input(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    old_node: dace_nodes.AccessNode,
    new_node: dace_nodes.AccessNode,
    new_node_offsets: Sequence[dace_sym.SymbolicType],
) -> None:
    reconfigured_neighbour: set[tuple[dace_nodes.Node, str | None]] = set()

    for producer_edge in list(state.in_edges(old_node)):
        producer: dace_nodes.Node = producer_edge.src
        producer_conn = producer_edge.src_conn
        new_producer_edge = gtx_transformations.utils.reroute_edge(
            is_producer_edge=True,
            current_edge=producer_edge,
            ss_offset=new_node_offsets,
            state=state,
            sdfg=sdfg,
            old_node=old_node,
            new_node=new_node,
        )
        if (producer, producer_conn) not in reconfigured_neighbour:
            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                is_producer_edge=True,
                new_edge=new_producer_edge,
                sdfg=sdfg,
                state=state,
                ss_offset=new_node_offsets,
                old_node=old_node,
                new_node=new_node,
            )
            reconfigured_neighbour.add((producer, producer_conn))

    for consumer_edge in list(state.out_edges(old_node)):
        consumer: dace_nodes.Node = consumer_edge.dst
        consumer_conn = consumer_edge.dst_conn
        new_consumer_edge = gtx_transformations.utils.reroute_edge(
            is_producer_edge=False,
            current_edge=consumer_edge,
            ss_offset=new_node_offsets,
            state=state,
            sdfg=sdfg,
            old_node=old_node,
            new_node=new_node,
        )
        if (consumer, consumer_conn) not in reconfigured_neighbour:
            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                is_producer_edge=False,
                new_edge=new_consumer_edge,
                sdfg=sdfg,
                state=state,
                ss_offset=new_node_offsets,
                old_node=old_node,
                new_node=new_node,
            )
            reconfigured_neighbour.add((consumer, consumer_conn))

    state.remove_node(old_node)
    sdfg.remove_data(old_node.data, validate=gtx_config.DEBUG)

    gtx_transformations.gt_propagate_strides_from_access_node(
        sdfg=sdfg,
        state=state,
        outer_node=new_node,
    )


def _cleanup_global_self_update(
    *,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    if_stmt_node: dace_nodes.NestedSDFG,
    if_stmt_else_state: dace.SDFGState,
    if_stmt_output: str,
    scan_node: dace_nodes.NestedSDFG,
    scan_compute_st: dace.SDFGState,
    compute_src_node: dace_nodes.AccessNode,
    compute_dst_node: dace_nodes.AccessNode,
):
    scan_sdfg = scan_node.sdfg
    assert isinstance(compute_dst_node.desc(scan_sdfg), dace.data.Scalar)

    # retrieve the source data outside the scan map scope
    assert len(list(state.in_edges_by_connector(scan_node, compute_src_node.data))) == 1
    top_level_input_edge = next(state.in_edges_by_connector(scan_node, compute_src_node.data))
    assert isinstance(top_level_input_edge.src, dace_nodes.MapEntry)
    map_entry_in_edge_conn = "IN_" + top_level_input_edge.src_conn[4:]
    assert (
        len(list(state.in_edges_by_connector(top_level_input_edge.src, map_entry_in_edge_conn)))
        == 1
    )
    top_level_src_node = next(
        state.in_edges_by_connector(top_level_input_edge.src, map_entry_in_edge_conn)
    ).src
    assert isinstance(top_level_src_node, dace_nodes.AccessNode)
    assert top_level_src_node.desc(sdfg).transient

    # retrieve the outer write to destination buffer in the compute state
    assert scan_compute_st.out_degree(compute_dst_node) == 1
    scan_output_edge = scan_compute_st.out_edges(compute_dst_node)[0]
    assert isinstance(scan_output_edge.dst, dace_nodes.AccessNode)
    output_node = scan_output_edge.dst
    output_desc = output_node.desc(scan_sdfg)
    assert not output_desc.transient
    output_data = output_node.data
    assert output_data.startswith("__gtir_scan_output")
    output_subset = scan_output_edge.data.get_dst_subset(scan_output_edge, scan_compute_st)
    assert output_subset.num_elements() == 1

    if_stmt_sdfg = if_stmt_node.sdfg
    if_stmt_global_output, _ = if_stmt_sdfg.add_scalar(
        output_data, output_desc.dtype, find_new_name=True
    )
    if_stmt_node.add_out_connector(if_stmt_global_output)

    for if_stmt_state in if_stmt_sdfg.states():
        if if_stmt_state != if_stmt_else_state:
            sink_nodes = [
                node for node in if_stmt_state.sink_nodes() if node.data == if_stmt_output
            ]
            assert len(sink_nodes) <= 1
            if sink_nodes:
                local_output_node = sink_nodes[0]
                assert if_stmt_state.in_degree(local_output_node) == 1
                if_stmt_output_edge = if_stmt_state.in_edges(local_output_node)[0]
                src_subset = if_stmt_output_edge.data.get_src_subset(
                    if_stmt_output_edge, if_stmt_state
                )
                if_stmt_state.add_edge(
                    if_stmt_output_edge.src,
                    if_stmt_output_edge.src_conn,
                    if_stmt_state.add_access(if_stmt_global_output),
                    None,
                    dace.Memlet(data=if_stmt_global_output, subset="0", other_subset=src_subset),
                )

    scan_compute_st.add_edge(
        if_stmt_node,
        if_stmt_global_output,
        output_node,
        None,
        dace.Memlet(data=output_data, subset=output_subset),
    )
    scan_compute_st.remove_edge(scan_output_edge)

    # retrieve the destination data outside the scan map scope
    assert len(list(state.out_edges_by_connector(scan_node, output_data))) == 1
    map_exit_in_edge = next(state.out_edges_by_connector(scan_node, output_data))
    assert isinstance(map_exit_in_edge.dst, dace_nodes.MapExit)
    map_exit_out_edge_conn = "OUT_" + map_exit_in_edge.dst_conn[3:]
    assert (
        len(list(state.out_edges_by_connector(map_exit_in_edge.dst, map_exit_out_edge_conn))) == 1
    )
    map_exit_out_edge = next(
        state.out_edges_by_connector(map_exit_in_edge.dst, map_exit_out_edge_conn)
    )
    top_level_dst_node = map_exit_out_edge.dst
    assert isinstance(top_level_dst_node, dace_nodes.AccessNode)
    top_level_dst_node_subset = map_exit_out_edge.data.get_dst_subset(map_exit_out_edge, state)

    # replace the source node outside the scan map with the output node
    new_top_level_src_node = state.add_access(top_level_dst_node.data)
    _replace_scan_input(
        sdfg=sdfg,
        state=state,
        old_node=top_level_src_node,
        new_node=new_top_level_src_node,
        new_node_offsets=top_level_dst_node_subset.min_element(),
    )
    print(
        f"Removed self-copy in {if_stmt_node.label}: {top_level_src_node.data} -> {compute_src_node.data} -> {compute_dst_node.data} -> {output_data} -> {top_level_dst_node.data}"
    )


def _graupel_run_self_copy_removal_inside_if_stmt(  # noqa: PLR0912, PLR0915 [too-many-branches, too-many-statements]
    *,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    scan_node: dace_nodes.NestedSDFG,
    scan_compute_st: dace.SDFGState,
    scan_update_st: dace.SDFGState,
    if_stmt_node: dace_nodes.NestedSDFG,
) -> None:
    scan_sdfg = scan_node.sdfg
    nsdfg = if_stmt_node.sdfg
    assert len(nsdfg.nodes()) == 1 and isinstance(
        nsdfg.nodes()[0], dace_sdfg.state.ConditionalBlock
    )
    if_region = nsdfg.nodes()[0]
    assert len(list(br[1] for br in if_region.branches if br[0] is None)) == 1
    else_br = next(br[1] for br in if_region.branches if br[0] is None)
    assert isinstance(else_br.start_block, dace.SDFGState)
    assert len(if_region.out_degree(else_br.start_block)) == 0
    else_st = else_br.start_block
    src_nodes = [node for node in else_st.source_nodes() if isinstance(node, dace_nodes.AccessNode)]

    for src_node in src_nodes:
        assert not src_node.desc(nsdfg).transient
        if else_st.out_degree(src_node) != 1:
            continue
        self_copy_edge = else_st.out_edges(src_node)[0]
        dst_node = self_copy_edge.dst
        if else_st.out_degree(dst_node) != 0:
            continue
        assert not dst_node.desc(nsdfg).transient

        # retrieve the source data to copy in the compute state
        assert len(list(scan_compute_st.in_edges_by_connector(if_stmt_node, src_node.data))) == 1
        compute_read_edge = next(scan_compute_st.in_edges_by_connector(if_stmt_node, src_node.data))
        compute_src_node = compute_read_edge.src
        assert isinstance(compute_src_node, dace_nodes.AccessNode)

        # retrieve the destination node in the compute state, where the data is written
        assert len(list(scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data))) == 1
        compute_write_edge = next(
            scan_compute_st.out_edges_by_connector(if_stmt_node, dst_node.data)
        )
        compute_dst_node = compute_write_edge.dst
        assert (
            isinstance(compute_dst_node, dace_nodes.AccessNode)
            and scan_compute_st.in_degree(compute_dst_node) == 1
        )
        output_data_name = compute_dst_node.data
        scan_update_last_level = scan_node.sdfg.nodes()[-1]
        assert "scan_last_level" in scan_update_last_level.label
        # If only the last level of the output data is updated then don't consider it as a self-copy as we only write in the end only its last level
        if any(
            node
            for node in scan_update_last_level.nodes()
            if isinstance(node, dace_nodes.AccessNode) and node.data == compute_dst_node.data
        ):
            continue

        # retrieve the data access inside the scan update state
        update_src_nodes = [
            node for node in scan_update_st.source_nodes() if node.data == output_data_name
        ]
        update_dst_nodes = [
            node
            for node in scan_compute_st.sink_nodes()
            if node.data == output_data_name and not node.desc(scan_compute_st).transient
        ]
        assert (len(update_src_nodes) <= 1 and len(update_dst_nodes) == 0) or (
            len(update_src_nodes) == 0 and len(update_dst_nodes) <= 1
        )
        if not update_src_nodes and not update_dst_nodes:
            continue
        if update_src_nodes:
            update_src_node = update_src_nodes[0]
            assert scan_update_st.out_degree(update_src_node) == 1
            update_write_edge = scan_update_st.out_edges(update_src_node)[0]
            update_dst_node = update_write_edge.dst
            assert isinstance(update_dst_node, dace_nodes.AccessNode)
            assert (
                scan_update_st.in_degree(update_dst_node) == 1
                and scan_update_st.out_degree(update_dst_node) == 0
            )
        else:
            update_dst_node = update_dst_nodes[0]
            assert scan_compute_st.in_degree(update_dst_node) == 1
            assert scan_compute_st.out_degree(update_dst_node) == 0

        if compute_src_node.desc(scan_sdfg).transient:  # Handles some scalar variables
            _cleanup_local_self_update(
                scan_sdfg=scan_sdfg,
                if_stmt_node=if_stmt_node,
                if_stmt_conn=dst_node.data,
                compute_src_node=compute_src_node,
                compute_dst_node=compute_dst_node,
                update_src_node=update_src_node,
                update_dst_node=update_dst_node,
                scan_compute_st=scan_compute_st,
                scan_update_st=scan_update_st,
            )
            else_st.remove_nodes_from([src_node, dst_node])
        elif update_src_nodes:  # handles `q_out_*` AccessNodes
            _cleanup_global_self_update(
                sdfg=sdfg,
                state=state,
                if_stmt_node=if_stmt_node,
                if_stmt_else_state=else_st,
                scan_node=scan_node,
                scan_compute_st=scan_compute_st,
                if_stmt_output=dst_node.data,
                compute_src_node=compute_src_node,
                compute_dst_node=compute_dst_node,
            )
        else:  # Handles `t_out`. `update_dst_nodes`` is not empty
            # replace the input and propagate the changes to all the edges
            # remove the copy in the false branch
            if update_dst_node.desc(scan_node.sdfg).transient:
                continue
            self_copy_edge_src_data = self_copy_edge.src.data
            map_entry_in_edge = next(
                state.in_edges_by_connector(scan_node, self_copy_edge_src_data)
            )
            # Means that there is no other computation before
            if not isinstance(map_entry_in_edge.src, dace_nodes.MapEntry):
                continue
            outer_data_map_out_connector = map_entry_in_edge.src_conn
            outer_access_node = next(
                state.in_edges_by_connector(
                    map_entry_in_edge.src, "IN_" + outer_data_map_out_connector[4:]
                )
            ).src
            map_exit_in_edge = next(state.out_edges_by_connector(scan_node, output_data_name))
            # Means that there is no computation with this after the scan
            if not isinstance(map_exit_in_edge.dst, dace_nodes.MapExit):
                continue
            outer_data_map_in_connector = map_exit_in_edge.dst_conn
            outer_dst_node = next(
                state.out_edges_by_connector(
                    map_exit_in_edge.dst, "OUT_" + outer_data_map_in_connector[3:]
                )
            ).dst
            # We just output to the AccessNode
            if not isinstance(outer_dst_node, dace_nodes.AccessNode):
                continue
            new_in_access_node = state.add_access(outer_dst_node.data)
            map_exit_out_edge = next(
                state.out_edges_by_connector(
                    map_exit_in_edge.dst, "OUT_" + map_exit_in_edge.dst_conn[3:]
                )
            )
            _replace_scan_input(
                sdfg=sdfg,
                state=state,
                old_node=outer_access_node,
                new_node=new_in_access_node,
                new_node_offsets=map_exit_out_edge.data.get_dst_subset(
                    map_exit_out_edge, state
                ).min_element(),
            )
            else_st.remove_nodes_from([self_copy_edge.src, self_copy_edge.dst])
            print(
                f"Removed self-copy in {if_stmt_node.label}: {outer_access_node.data} -> {src_node.data} -> {dst_node.data} -> {output_data_name} -> {outer_dst_node.data}"
            )

    if else_st.is_empty():
        if_region.remove_branch(else_br)


def remove_self_copy_inside_scan(sdfg: dace.SDFG) -> None:
    assert len(sdfg.states()) == 1
    st = sdfg.states()[0]
    assert (
        len(
            list(
                node
                for node in st.nodes()
                if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("scan_")
            )
        )
        == 1
    )
    scan_nsdfg_node = next(
        node
        for node in st.nodes()
        if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("scan_")
    )
    scan_sdfg = scan_nsdfg_node.sdfg
    assert len(scan_sdfg.nodes()) == 3
    assert isinstance(scan_sdfg.nodes()[1], dace_sdfg.state.LoopRegion)
    loop_regions = [
        scan_sdfg_node
        for scan_sdfg_node in scan_sdfg.nodes()
        if isinstance(scan_sdfg_node, dace_sdfg.state.LoopRegion)
    ]
    assert len(loop_regions) == 1
    scan_loop = loop_regions[0]
    assert len(scan_loop.nodes()) == 2 and all(
        isinstance(node, dace.SDFGState) for node in scan_loop.nodes()
    )
    if scan_loop.nodes()[0].label.startswith("scan_compute"):
        assert scan_loop.nodes()[1].label.startswith("scan_update")
        scan_compute_st, scan_update_st = scan_loop.nodes()
    else:
        assert scan_loop.nodes()[0].label.startswith("scan_update")
        scan_update_st, scan_compute_st = scan_loop.nodes()

    if_stmt_nodes = [
        node
        for node in scan_compute_st.nodes()
        if isinstance(node, dace_nodes.NestedSDFG) and node.label.startswith("if_stmt_")
    ]
    for if_stmt_node in if_stmt_nodes:
        _graupel_run_self_copy_removal_inside_if_stmt(
            sdfg=sdfg,
            state=st,
            scan_node=scan_nsdfg_node,
            scan_compute_st=scan_compute_st,
            scan_update_st=scan_update_st,
            if_stmt_node=if_stmt_node,
        )

    for input_access_nodes in ["te", "q_in_2", "q_in_3", "q_in_4", "q_in_5"]:
        all_maps_with_accessnode_input = [
            node
            for node in st.nodes()
            if isinstance(node, dace_nodes.MapEntry)
            and f"IN_{input_access_nodes}" in node.in_connectors
        ]
        all_maps_with_accessnode_input_and_if_stmt = [
            map_with_if
            for map_with_if in all_maps_with_accessnode_input
            if any(
                isinstance(map_node, dace_nodes.NestedSDFG)
                and map_node.label.startswith("if_stmt_")
                for map_node in st.scope_subgraph(map_with_if).nodes()
            )
        ]
        assert len(all_maps_with_accessnode_input_and_if_stmt) == 1
        map_with_accessnode_input_and_if_stmt = all_maps_with_accessnode_input_and_if_stmt[0]
        nsdfg_if_stmt_with_accessnode = next(
            node
            for node in st.scope_subgraph(map_with_accessnode_input_and_if_stmt).nodes()
            if isinstance(node, dace_nodes.NestedSDFG)
        )
        nsdfg_conditional_block = nsdfg_if_stmt_with_accessnode.sdfg.nodes()[0]
        else_branch = nsdfg_conditional_block.branches[1][1]
        else_branch_state = else_branch.nodes()[0]
        assert len(nsdfg_if_stmt_with_accessnode.out_connectors.keys()) == 1
        nsdfg_if_stmt_out_connector = next(
            iter(nsdfg_if_stmt_with_accessnode.out_connectors.keys())
        )
        output_edges = [
            edge
            for edge in st.out_edges_by_connector(
                nsdfg_if_stmt_with_accessnode, nsdfg_if_stmt_out_connector
            )
        ]
        assert len(output_edges) == 1
        output_edge = output_edges[0]
        intermediate_an = output_edge.dst
        assert isinstance(intermediate_an, dace_nodes.AccessNode)
        out_edge_of_inter_an = st.out_edges(intermediate_an)[0]
        dst_out_edge_of_inter_an = out_edge_of_inter_an.dst
        assert isinstance(dst_out_edge_of_inter_an, dace_nodes.MapExit)
        out_edges_of_map_exit = [
            oedge_map_exit
            for oedge_map_exit in st.out_edges_by_connector(
                dst_out_edge_of_inter_an, "OUT_" + out_edge_of_inter_an.dst_conn[3:]
            )
        ]
        assert len(out_edges_of_map_exit) == 1
        out_edge_of_map_exit = out_edges_of_map_exit[0]
        dst_out_edge_of_map_exit = out_edge_of_map_exit.dst
        assert isinstance(dst_out_edge_of_map_exit, dace_nodes.AccessNode)
        new_memlet = dace.Memlet(
            data=out_edge_of_inter_an.data.data,
            subset=copy.deepcopy(out_edge_of_inter_an.data.subset),
            other_subset=copy.deepcopy(output_edge.data.subset),
        )
        new_output_edge = dace_transformation.helpers.redirect_edge(
            state=st,
            edge=output_edge,
            new_dst=dst_out_edge_of_inter_an,
            new_dst_conn=out_edge_of_inter_an.dst_conn,
            new_memlet=new_memlet,
        )
        new_output_edge.data.allow_oob = True
        st.remove_edge(out_edge_of_inter_an)
        st.remove_node(intermediate_an)
        sdfg.arrays.pop(intermediate_an.data)
        if (
            "else_body" in else_branch.name
            and len(else_branch_state.nodes()) == 2
            and all(isinstance(node, dace_nodes.AccessNode) for node in else_branch_state.nodes())
        ):
            print(
                f"Removed self-copy in {nsdfg_if_stmt_with_accessnode.label} for '{input_access_nodes}' by removing the else branch"
            )
            else_branch_state.sdfg.remove_nodes_from(
                [internal_node for internal_node in else_branch_state.nodes()]
            )
            nsdfg_conditional_block.remove_branch(else_branch)
    sdfg.validate()


def rename_intermediate_access_nodes(sdfg: dace.SDFG) -> None:
    assert len(sdfg.states()) == 1
    st: dace.SDFGState = sdfg.states()[0]
    access_node_renaming_dict = {
        "q_out_2": "q_in_2",
        "q_out_3": "q_in_3",
        "q_out_4": "q_in_4",
        "q_out_5": "q_in_5",
        "t_out": "te",
    }

    def _update_repl_impl(
        repl: dict[str, str],
        old_name: str,
        old_symbols: Sequence[Any],
        new_name: str,
        new_symbols: Sequence[Any],
    ) -> None:
        for old_sym, new_sym in zip(old_symbols, new_symbols):
            if old_sym == new_sym:
                continue
            old_ssym = str(old_sym)
            new_ssym = str(new_sym)
            if (old_ssym in repl) and (repl[old_ssym] != new_ssym):
                raise NotImplementedError("Found symbol conflict.")
            elif (old_ssym.isdigit() and (not new_ssym.isdigit())) or new_ssym in repl:
                raise NotImplementedError()

            if old_sym.is_symbol:
                # The entry is a simple symbol, i.e. `stride_0_of_array_a`.
                repl[old_ssym] = new_ssym
            else:
                # The entry is a composed symbol, i.e. `range_end - range_start`.
                #  This mostly happens for the shapes, we require that also the
                #  new symbol is a composed symbol and we perform "fancy" renaming.
                old_sfsyms = [str(fs) for fs in old_sym.free_symbols]
                new_sfsyms = {str(fs) for fs in new_sym.free_symbols}
                if len(old_sfsyms) != len(new_sfsyms):
                    raise NotImplementedError()
                for old_sfsym in old_sfsyms:
                    if old_name not in old_sfsym:
                        raise NotImplementedError()
                    assert new_name not in old_sfsym
                    new_sfsym = old_sfsym.replace(old_name, new_name)
                    if new_sfsym not in new_sfsyms:
                        raise NotImplementedError()
                    if (old_sfsym in repl) and (repl[old_sfsym] != new_sfsym):
                        raise NotImplementedError()
                    repl[old_sfsym] = new_sfsym
                    new_sfsyms.discard(new_sfsym)

    def _update_repl(
        sdfg: dace.SDFG,
        repl: dict[str, str],
        old_name: str,
        new_name: str,
    ) -> None:
        old_desc = sdfg.arrays[old_name]
        new_desc = sdfg.arrays[new_name]
        for what in ["shape", "strides"]:
            _update_repl_impl(
                repl,
                old_name=old_name,
                old_symbols=getattr(old_desc, what),
                new_name=new_name,
                new_symbols=getattr(new_desc, what),
            )

    repl: dict[str, str] = {}
    for old_name, new_name in access_node_renaming_dict.items():
        _update_repl(sdfg=sdfg, repl=repl, old_name=old_name, new_name=new_name)
        print(
            f"Renaming '{old_name}' to '{new_name}' in the SDFG because they refer to the same data."
        )

    for dnode in st.data_nodes():
        if dnode.data in access_node_renaming_dict:
            old_data = dnode.data
            new_data = access_node_renaming_dict[old_data]
            dnode.data = new_data
    for edge in st.edges():
        if edge.data.data in access_node_renaming_dict:
            old_data = edge.data.data
            new_data = access_node_renaming_dict[old_data]
            edge.data.data = access_node_renaming_dict[edge.data.data]

    if repl:
        sdfg.replace_dict(
            repldict=repl,
            replace_keys=False,  # This will keep the old descriptor in the graph.
            replace_in_graph=True,
        )

    sdfg.validate()
