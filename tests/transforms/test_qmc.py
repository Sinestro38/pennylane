# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the quantum_monte_carlo transform"""
import itertools

import numpy as np

import pennylane as qml
from pennylane.transforms.qmc import _apply_controlled_z
from pennylane.templates.subroutines.qmc import _make_Z


def test_apply_controlled_z():
    """Test if the _apply_controlled_z performs the correct transformation by reconstructing the
    unitary and comparing against the one provided in _make_Z."""
    n_wires = 2
    n_all_wires = n_wires + 1

    wires = range(n_wires)
    control_wire = n_wires
    work_wires = None

    dev = qml.device("default.qubit", wires=range(n_all_wires))

    @qml.qnode(dev)
    def unitary_z(basis_state):
        qml.BasisState(basis_state, wires=range(n_all_wires))
        _apply_controlled_z(wires=wires, control_wire=control_wire, work_wires=work_wires)
        return qml.state()

    bitstrings = list(itertools.product([0, 1], repeat=n_all_wires))
    u = [unitary_z(bitstring).numpy() for bitstring in bitstrings]
    u = np.array(u).T

    # Note the sign flip in the following. The sign does not matter when performing the Q unitary
    # because two Zs are used.
    z_ideal = -_make_Z(2 ** n_wires)

    @qml.qnode(dev)
    def unitary_z_ideal(basis_state):
        qml.BasisState(basis_state, wires=range(n_all_wires))
        qml.ControlledQubitUnitary(z_ideal, wires=wires, control_wires=control_wire)
        return qml.state()

    u_ideal = [unitary_z_ideal(bitstring).numpy() for bitstring in bitstrings]
    u_ideal = np.array(u_ideal).T

    assert np.allclose(u, u_ideal)
