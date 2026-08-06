"""
Quantum Tic-Tac-Toe  —  Human vs Quantum AI
=============================================
You are X (human).  The Quantum AI is O.

How the Quantum AI works
------------------------
The AI uses a real Qiskit quantum circuit to decide its next move:

1. Each empty cell is mapped to a qubit.
2. All qubits start in equal superposition via Hadamard gates (H).
3. If the AI can win immediately, it applies a phase-kick (Z gate) on that
   winning cell's qubit, then an inversion-about-average (Grover diffusion)
   to amplify that cell's probability — making it the likely choice.
4. If no immediate win exists but the human threatens a win, the same
   amplification targets the blocking cell.
5. Otherwise all cells stay in equal superposition and the AI picks randomly
   from the measurement outcome.
6. The circuit is run for `shots=1` on AerSimulator; the measured qubit
   index maps back to an actual board cell.

Additionally, the AI sometimes plays a *quantum / superposition* move:
it places a "?" mark on two cells simultaneously (entangled pair using a
CNOT after H).  When that entangled pair is later measured, one cell gets O
and the other is left empty — adding genuine quantum uncertainty to the game.

How to play
-----------
  - The board has 9 cells numbered 1-9.
  - On your turn enter a cell number (1-9).
  - Type 'quit' to exit.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random

# ── board helpers ─────────────────────────────────────────────────────────────

def fresh_board():
    """9-cell board (1-indexed; index 0 unused)."""
    return [None] * 10

def display_board(board, entangled_cells):
    """Pretty-print the 3×3 board. Entangled AI cells show as '⊗'."""
    symbols = []
    for i in range(1, 10):
        if board[i] is not None:
            symbols.append(board[i])
        elif i in entangled_cells:
            symbols.append("⊗")
        else:
            symbols.append(str(i))

    print("\n")
    print(f" {symbols[0]} | {symbols[1]} | {symbols[2]} ")
    print("---+---+---")
    print(f" {symbols[3]} | {symbols[4]} | {symbols[5]} ")
    print("---+---+---")
    print(f" {symbols[6]} | {symbols[7]} | {symbols[8]} ")
    print()

WIN_LINES = [
    (1, 2, 3), (4, 5, 6), (7, 8, 9),   # rows
    (1, 4, 7), (2, 5, 8), (3, 6, 9),   # cols
    (1, 5, 9), (3, 5, 7),              # diagonals
]

def check_winner(board):
    """Return 'X', 'O', or None."""
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None

def empty_cells(board, entangled_cells):
    """Cells that are neither classically filled nor entangled."""
    return [i for i in range(1, 10)
            if board[i] is None and i not in entangled_cells]

def board_full(board, entangled_cells):
    return all(board[i] is not None or i in entangled_cells
               for i in range(1, 10))

# ── quantum circuit helpers ───────────────────────────────────────────────────

def _grover_diffusion(qc, qubits):
    """
    Inversion-about-average on the given qubits.
    Amplifies the phase-kicked qubit's amplitude.
    """
    qc.h(qubits)
    qc.x(qubits)
    # Multi-controlled Z via H + MCX + H on last qubit
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def quantum_pick_cell(free_cells, board):
    """
    Use a Qiskit circuit to choose a cell from `free_cells`.

    Strategy (in priority order):
      1. Immediate win  → amplify that cell with Grover diffusion.
      2. Block human win → amplify that cell with Grover diffusion.
      3. No threat      → equal superposition, random measurement.

    Returns the chosen cell index (1-9).
    """
    n = len(free_cells)
    if n == 1:
        return free_cells[0]

    # --- find strategic target cell (win or block) ---
    target_local = None   # index into free_cells list

    def _find_threat(mark):
        """Return local index of a cell that would complete a line for `mark`."""
        for a, b, c in WIN_LINES:
            line = [a, b, c]
            owned = [x for x in line if board[x] == mark]
            empty = [x for x in line if board[x] is None]
            if len(owned) == 2 and len(empty) == 1 and empty[0] in free_cells:
                return free_cells.index(empty[0])
        return None

    target_local = _find_threat("O")          # can AI win?
    if target_local is None:
        target_local = _find_threat("X")      # must AI block?

    # --- build quantum circuit ---
    qc = QuantumCircuit(n, n)

    # Equal superposition over all free cells
    qc.h(list(range(n)))

    if target_local is not None and n >= 2:
        # Phase-kick the target qubit, then amplify via Grover diffusion
        qc.z(target_local)
        _grover_diffusion(qc, list(range(n)))

    qc.measure(list(range(n)), list(range(n)))

    simulator = AerSimulator()
    job = simulator.run(qc, shots=1)
    counts = job.result().get_counts(qc)
    bitstring = list(counts.keys())[0]

    # Count the number of |1⟩ qubits; pick the first one found.
    # Qiskit bitstring: rightmost char = qubit 0.
    chosen_local = None
    for idx in range(n):
        if bitstring[-(idx + 1)] == "1":
            chosen_local = idx
            break

    # Fallback: if all qubits measured 0 (rare), pick target or random
    if chosen_local is None:
        chosen_local = target_local if target_local is not None else 0

    return free_cells[chosen_local]


def quantum_entangled_move(free_cells, board):
    """
    AI places an entangled (superposition) move across two cells using:
      - H on qubit 0  (superposition)
      - CNOT(0 → 1)   (entanglement)
      - Measure both

    Outcomes:
      |00⟩ → neither cell gets O  (AI skips this turn — unlikely but possible)
      |01⟩ → cell A gets O
      |10⟩ → cell B gets O
      |11⟩ → both get O  (double move — quantum advantage!)

    Returns (cell_a, cell_b, result_a, result_b) where result is 'O' or None.
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    simulator = AerSimulator()
    counts = simulator.run(qc, shots=1).result().get_counts(qc)
    bits = list(counts.keys())[0]   # rightmost = qubit 0
    bit0, bit1 = bits[-1], bits[-2]

    cell_a, cell_b = free_cells[0], free_cells[1]
    result_a = "O" if bit0 == "1" else None
    result_b = "O" if bit1 == "1" else None
    return cell_a, cell_b, result_a, result_b


# ── quantum AI turn ───────────────────────────────────────────────────────────

def quantum_ai_move(board, entangled_cells):
    """
    Decide and execute the Quantum AI's move.
    Has a 25 % chance of attempting an entangled (two-cell) move when 2+
    cells are free; otherwise picks a single cell via Grover amplification.
    """
    free = empty_cells(board, entangled_cells)

    if not free:
        return   # board full

    # 25 % chance of entangled move when enough free cells exist
    if len(free) >= 2 and random.random() < 0.25:
        random.shuffle(free)
        cell_a, cell_b, res_a, res_b = quantum_entangled_move(free, board)
        print(f"\n  ⚛  Quantum AI plays an ENTANGLED move on cells {cell_a} & {cell_b}!")
        print(f"     Running Bell-state circuit (H + CNOT) …")
        placed = []
        if res_a == "O":
            board[cell_a] = "O"
            placed.append(cell_a)
        if res_b == "O":
            board[cell_b] = "O"
            placed.append(cell_b)
        if placed:
            print(f"     Collapsed → O lands on cell(s): {placed}")
        else:
            print("     Collapsed → both qubits |0⟩ — AI loses this turn (quantum chance)!")
    else:
        # Single-cell move with Grover-amplified selection
        cell = quantum_pick_cell(free, board)
        board[cell] = "O"
        print(f"\n  ⚛  Quantum AI (O) plays cell {cell}  [Grover-amplified selection]")


# ── human turn ────────────────────────────────────────────────────────────────

def human_move(board, entangled_cells):
    """Prompt the human player until a valid cell is entered."""
    free = empty_cells(board, entangled_cells)
    while True:
        raw = input("Your move (X) ▶  ").strip().lower()

        if raw == "quit":
            return "quit"

        if not raw.isdigit():
            print("  Enter a number 1-9, or 'quit'.")
            continue

        cell = int(raw)
        if cell < 1 or cell > 9:
            print("  Cell must be between 1 and 9.")
            continue

        if cell not in free:
            print(f"  Cell {cell} is already occupied. Free cells: {free}")
            continue

        board[cell] = "X"
        return cell


# ── main game loop ────────────────────────────────────────────────────────────

def print_instructions():
    print("=" * 56)
    print("     QUANTUM TIC-TAC-TOE  —  Human (X)  vs  Quantum AI (O)")
    print("=" * 56)
    print("  You are X.  Enter a cell number (1-9) to place your mark.")
    print("  The Quantum AI uses real Qiskit circuits to choose its move.")
    print("  ⊗  = AI entangled mark (superposition, not yet collapsed)")
    print("  Type 'quit' to exit.")
    print("=" * 56)
    print()


def play():
    print_instructions()
    board = fresh_board()
    entangled_cells = set()   # cells the AI claimed but left in superposition
    turn = 0                  # even = human (X), odd = AI (O)

    while True:
        display_board(board, entangled_cells)

        # ── check game-over before next move ──────────────────────────────
        winner = check_winner(board)
        if winner:
            print(f"{'🎉  You win! Well played!' if winner == 'X' else '🤖  Quantum AI wins!'}")
            break
        if board_full(board, entangled_cells):
            print("It's a draw!")
            break

        # ── human turn ────────────────────────────────────────────────────
        if turn % 2 == 0:
            print("Your turn (X)")
            result = human_move(board, entangled_cells)
            if result == "quit":
                print("Thanks for playing! Goodbye.")
                break

        # ── quantum AI turn ───────────────────────────────────────────────
        else:
            print("Quantum AI (O) is thinking…")
            quantum_ai_move(board, entangled_cells)

        turn += 1

    # Final board
    display_board(board, entangled_cells)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    play()
