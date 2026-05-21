# ── Imports ───────────────────────────────────────────────────────────────────
import gradio as gr
import csv
import io
import math
import inspect
import datetime
import tempfile
import os

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D

import pickle

try:
    with open("saved_models/test_reactions.pkl", "rb") as f:
        test_reactions = pickle.load(f)
except Exception:
    test_reactions = []
# ─────────────────────────────────────────────────────────────────────────────
#  Reaction type classifier  (SMARTS-based heuristics)
# ─────────────────────────────────────────────────────────────────────────────

REACTION_TYPES = [
    ("Suzuki coupling",        ["[c:1][Br,I,Cl]", "[B]([OH])[OH]"]),
    ("Buchwald-Hartwig",       ["[c:1][Br,I]",    "[NH2,NH1]"]),
    ("Heck reaction",          ["[c:1][Br,I]",    "[CH2]=[CH]"]),
    ("Esterification",         ["[CX3](=O)[OH]",  "[OX2H]"]),
    ("Amide coupling",         ["[CX3](=O)[OH]",  "[NX3;H1,H2]"]),
    ("Reductive amination",    ["[CX3H1]=O",      "[NX3;H1,H2]"]),
    ("Grignard reaction",      ["[Mg][Br,Cl,I]",  "[CX3]=O"]),
    ("Wittig reaction",        ["[P+]([C])",       "[CX3]=O"]),
    ("Diels-Alder",            ["[CX3]=[CX3]-[CX3]=[CX3]", "[CX3]=[CX3]"]),
    ("Nucleophilic substitution", ["[C][Br,Cl,I,F]", "[OH,NH2,SH]"]),
    ("Aldol condensation",     ["[CX3H1]=O",      "[CH3]C=O"]),
    ("Michael addition",       ["[CX3]=[CX3]C=O", "[NH,OH,CH]"]),
    ("Oxidation",              ["[OX2H]",         "[O]"]),
    ("Reduction",              ["C=O",            "[H][H]"]),
]

def classify_reaction(reactant_smiles: str) -> str:
    """Heuristic reaction type label from reactant SMILES."""
    parts = [p.strip() for p in reactant_smiles.split(".")]
    mols  = [Chem.MolFromSmiles(p) for p in parts if Chem.MolFromSmiles(p)]
    if not mols:
        return "Unknown"
    for name, smarts_list in REACTION_TYPES:
        hits = 0
        for smarts in smarts_list:
            patt = Chem.MolFromSmarts(smarts)
            if patt and any(m.HasSubstructMatch(patt) for m in mols):
                hits += 1
        if hits >= len(smarts_list):
            return name
        elif hits == 1 and len(smarts_list) == 1:
            return name
    return "General organic"


# ─────────────────────────────────────────────────────────────────────────────
#  Tanimoto similarity (Morgan fingerprints)
# ─────────────────────────────────────────────────────────────────────────────

def tanimoto_similarity(smi1: str, smi2: str) -> float:
    """Morgan FP Tanimoto similarity between two SMILES. Returns -1 on error."""
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    if m1 is None or m2 is None:
        return -1.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius=2, nBits=2048)
    return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  SMILES diff  (character-level coloured HTML)
# ─────────────────────────────────────────────────────────────────────────────

def smiles_diff_html(pred: str, actual: str) -> str:
    """Return HTML with green (match) / red (mismatch) character colouring."""
    if not pred or not actual:
        return "<i style='color:#888'>Supply both SMILES to see diff.</i>"
    html = ["<div style='font-family:monospace;font-size:13px;line-height:1.8;word-break:break-all'>"]
    html.append("<b>Predicted:</b> ")
    for i, ch in enumerate(pred):
        if i < len(actual) and ch == actual[i]:
            html.append(f"<span style='color:#2a9d5c'>{ch}</span>")
        else:
            html.append(f"<span style='color:#e63946;text-decoration:underline'>{ch}</span>")
    if len(pred) < len(actual):
        html.append(f"<span style='color:#aaa'>[+{len(actual)-len(pred)} chars missing]</span>")
    html.append("<br><b>Actual:   </b> ")
    for i, ch in enumerate(actual):
        if i < len(pred) and ch == pred[i]:
            html.append(f"<span style='color:#2a9d5c'>{ch}</span>")
        else:
            html.append(f"<span style='color:#e63946;text-decoration:underline'>{ch}</span>")
    if len(actual) < len(pred):
        html.append(f"<span style='color:#aaa'>[+{len(pred)-len(actual)} extra chars]</span>")
    html.append("</div>")
    return "".join(html)


# ─────────────────────────────────────────────────────────────────────────────
#  Molecule drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_svg(smiles: str, width: int = 300, height: int = 200) -> str:
    if not smiles or not smiles.strip():
        return _placeholder_svg("(no SMILES)", width, height)
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return _placeholder_svg("Invalid SMILES", width, height)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _placeholder_svg(msg: str, w: int, h: int) -> str:
    return (
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{w}" height="{h}" fill="#f0f0f0" rx="8"/>'
        f'<text x="{w//2}" y="{h//2}" text-anchor="middle" '
        f'font-family="monospace" font-size="13" fill="#888">{msg}</text>'
        f'</svg>'
    )


def smiles_html(label: str, smiles: str, width: int = 260, height: int = 180) -> str:
    svg = smiles_to_svg(smiles, width, height)
    display_smi = (smiles[:60] + "…") if smiles and len(smiles) > 60 else (smiles or "—")
    return (
        f'<div style="text-align:center;padding:8px;min-width:{width}px">'
        f'<p style="font-size:12px;color:#666;margin-bottom:4px;font-weight:600">{label}</p>'
        f'{svg}'
        f'<p style="font-size:10px;color:#999;margin-top:4px;font-family:monospace;'
        f'word-break:break-all">{display_smi}</p>'
        f'</div>'
    )


def side_by_side_html(reactant: str, predicted: str, actual: str = "",
                      rxn_type: str = "") -> str:
    arrow = ('<div style="font-size:24px;color:#aaa;display:flex;'
             'align-items:center;padding:0 6px">→</div>')
    badge = ""
    if rxn_type:
        badge = (f'<div style="text-align:center;margin-bottom:8px">'
                 f'<span style="background:#e8f4fd;color:#1565c0;font-size:12px;'
                 f'padding:3px 10px;border-radius:12px;font-weight:500">'
                 f'🔬 {rxn_type}</span></div>')
    cols = (f'<div style="overflow-x:auto">{badge}'
            f'<div style="display:flex;align-items:center;flex-wrap:nowrap;gap:2px">'
            f'{smiles_html("Reactant", reactant)}{arrow}'
            f'{smiles_html("Predicted product", predicted)}')
    if actual.strip():
        cols += f'{arrow}{smiles_html("Actual product", actual)}'
    cols += '</div></div>'
    return cols


# ─────────────────────────────────────────────────────────────────────────────
#  Molecular property helpers
# ─────────────────────────────────────────────────────────────────────────────

FUNCTIONAL_GROUP_SMARTS = {
    "ester":           "[CX3](=O)[OX2H0]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "aldehyde":        "[CX3H1](=O)",
    "ketone":          "[CX3](=O)[#6]",
    "primary amine":   "[NX3;H2]",
    "secondary amine": "[NX3;H1]",
    "amide":           "[NX3][CX3](=O)",
    "alcohol":         "[OX2H]",
    "ether":           "[OD2]([#6])[#6]",
    "halide":          "[F,Cl,Br,I]",
    "nitrile":         "[NX1]#[CX2]",
    "aromatic ring":   "c1ccccc1",
    "alkyne":          "[CX2]#[CX2]",
    "alkene":          "[CX3]=[CX3]",
    "sulfonamide":     "[SX4](=O)(=O)[NX3]",
    "sulfone":         "[SX4](=O)(=O)",
    "phosphate":       "[PX4](=O)([OH])[OH]",
    "thiol":           "[SX2H]",
    "epoxide":         "[OX2r3]",
    "anhydride":       "[CX3](=O)[OX2][CX3]=O",
}

def detect_functional_groups(mol) -> list:
    found = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            found.append(name)
    return found


def get_mol_properties(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES."
    formula  = rdMolDescriptors.CalcMolFormula(mol)
    mw       = round(Descriptors.MolWt(mol), 2)
    hba      = rdMolDescriptors.CalcNumHBA(mol)
    hbd      = rdMolDescriptors.CalcNumHBD(mol)
    rings    = rdMolDescriptors.CalcNumRings(mol)
    arom     = rdMolDescriptors.CalcNumAromaticRings(mol)
    rot      = rdMolDescriptors.CalcNumRotatableBonds(mol)
    logp     = round(Descriptors.MolLogP(mol), 2)
    tpsa     = round(Descriptors.TPSA(mol), 2)
    hac      = mol.GetNumHeavyAtoms()
    fg_list  = detect_functional_groups(mol)
    fg_str   = ", ".join(fg_list) if fg_list else "none detected"
    lipinski = all([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
    lip_str  = "✅ Pass" if lipinski else "❌ Fail"
    return (
        f"Formula          : {formula}\n"
        f"Mol. weight      : {mw} g/mol\n"
        f"Heavy atom count : {hac}\n"
        f"LogP             : {logp}\n"
        f"TPSA             : {tpsa} Å²\n"
        f"HB acceptors     : {hba}\n"
        f"HB donors        : {hbd}\n"
        f"Rings            : {rings}  (aromatic: {arom})\n"
        f"Rotatable bonds  : {rot}\n"
        f"Lipinski Ro5     : {lip_str}\n"
        f"Functional groups: {fg_str}"
    )


def validate_smiles(smiles: str) -> str:
    if not smiles or not smiles.strip():
        return "⚠️  No SMILES provided."
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return f"❌  Invalid SMILES\nRDKit could not parse: '{smiles}'"
    canonical = Chem.MolToSmiles(mol)
    props = get_mol_properties(smiles)
    return f"✅  Valid SMILES\nCanonical: {canonical}\n\n{props}"


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles


def reaction_smarts(reactant: str, product: str) -> str:
    """Generate reaction SMARTS string reactant>>product."""
    r_mol = Chem.MolFromSmiles(reactant)
    p_mol = Chem.MolFromSmiles(product)
    if r_mol is None or p_mol is None:
        return "Cannot generate — invalid SMILES."
    r_smi = Chem.MolToSmiles(r_mol)
    p_smi = Chem.MolToSmiles(p_mol)
    return f"{r_smi}>>{p_smi}"


# ─────────────────────────────────────────────────────────────────────────────
#  NEW FEATURE 1: Atom mapping
# ─────────────────────────────────────────────────────────────────────────────

def generate_atom_mapping_html(reactant_smiles: str, product_smiles: str) -> str:
    """
    Generate atom-mapped SVGs for reactant and product using RDKit's
    atom-map numbers. Attempts to use ReactionFromSmarts to propagate
    atom maps; falls back to MCS-based colouring if unavailable.
    """
    if not reactant_smiles.strip() or not product_smiles.strip():
        return "<i style='color:#888'>Supply both reactant and product SMILES to see atom mapping.</i>"

    r_mol = Chem.MolFromSmiles(reactant_smiles.strip())
    p_mol = Chem.MolFromSmiles(product_smiles.strip())
    if r_mol is None or p_mol is None:
        return "<i style='color:#e63946'>Invalid SMILES — cannot generate atom mapping.</i>"

    # Try to find common substructure and colour matched atoms
    try:
        from rdkit.Chem import rdFMCS
        mcs_result = rdFMCS.FindMCS(
            [r_mol, p_mol],
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
            timeout=3,
        )
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString) if mcs_result else None
    except Exception:
        mcs_mol = None

    # Build colour maps for matched atoms
    r_highlight = {}
    p_highlight = {}
    r_atom_list = []
    p_atom_list = []

    PALETTE = [
        (0.23, 0.71, 0.53),  # green
        (0.20, 0.53, 0.80),  # blue
        (0.92, 0.60, 0.15),  # orange
        (0.78, 0.24, 0.37),  # red
        (0.58, 0.38, 0.78),  # purple
        (0.15, 0.68, 0.68),  # teal
        (0.90, 0.40, 0.60),  # pink
        (0.60, 0.76, 0.25),  # lime
    ]

    if mcs_mol:
        r_match = r_mol.GetSubstructMatch(mcs_mol)
        p_match = p_mol.GetSubstructMatch(mcs_mol)
        for idx, (r_idx, p_idx) in enumerate(zip(r_match, p_match)):
            colour = PALETTE[idx % len(PALETTE)]
            r_highlight[r_idx] = colour
            p_highlight[p_idx] = colour
            r_atom_list.append(r_idx)
            p_atom_list.append(p_idx)

    def _draw_with_highlights(mol, highlight_atoms, highlight_map, label, width=280, height=200):
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        if highlight_map:
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=highlight_map,
                highlightBonds=[],
            )
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        smi_display = Chem.MolToSmiles(mol)
        smi_display = (smi_display[:55] + "…") if len(smi_display) > 55 else smi_display
        return (
            f'<div style="text-align:center;padding:6px">'
            f'<p style="font-size:12px;font-weight:600;color:#444;margin-bottom:4px">{label}</p>'
            f'{svg}'
            f'<p style="font-size:10px;color:#999;font-family:monospace;word-break:break-all;margin-top:4px">'
            f'{smi_display}</p></div>'
        )

    r_svg = _draw_with_highlights(r_mol, r_atom_list, r_highlight, "Reactant")
    p_svg = _draw_with_highlights(p_mol, p_atom_list, p_highlight, "Product")

    mcs_note = ""
    if mcs_mol:
        n_mapped = len(r_atom_list)
        mcs_note = (
            f'<p style="font-size:11px;color:#555;margin:6px 0 0">'
            f'🔗 {n_mapped} atoms mapped via MCS (Maximum Common Substructure). '
            f'Matching colours indicate corresponding atoms.</p>'
        )
    else:
        mcs_note = '<p style="font-size:11px;color:#888;margin:6px 0 0">No common substructure found.</p>'

    legend_items = ""
    for i in range(min(8, max(len(r_atom_list), 1))):
        r, g, b = PALETTE[i % len(PALETTE)]
        hex_col = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        if i < len(r_atom_list):
            r_idx = r_atom_list[i]
            p_idx = p_atom_list[i] if i < len(p_atom_list) else "—"
            legend_items += (
                f'<span style="display:inline-flex;align-items:center;gap:4px;'
                f'margin:2px 6px 2px 0;font-size:11px;font-family:monospace">'
                f'<span style="width:12px;height:12px;border-radius:3px;background:{hex_col};'
                f'display:inline-block"></span>'
                f'R:{r_idx}→P:{p_idx}</span>'
            )

    arrow = '<div style="font-size:26px;color:#aaa;display:flex;align-items:center;padding:0 10px">→</div>'
    html = (
        f'<div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:10px;padding:12px">'
        f'<div style="display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:4px">'
        f'{r_svg}{arrow}{p_svg}</div>'
        f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #eee">'
        f'{mcs_note}'
        f'<div style="margin-top:6px">{legend_items}</div>'
        f'</div></div>'
    )
    return html


# ─────────────────────────────────────────────────────────────────────────────
#  NEW FEATURE 2: Retrosynthesis mode
# ─────────────────────────────────────────────────────────────────────────────

# Common retrosynthetic disconnections: product SMARTS → (reactant1, reactant2) SMARTS
RETRO_RULES = [
    # Ester → carboxylic acid + alcohol
    (
        "[CX3:1](=O)[OX2H0:2]",
        "[CX3:1](=O)[OH]", "[OX2H:2]",
        "Ester hydrolysis → Carboxylic acid + Alcohol"
    ),
    # Amide → carboxylic acid + amine
    (
        "[CX3:1](=O)[NX3H:2]",
        "[CX3:1](=O)[OH]", "[NX3H2:2]",
        "Amide bond → Carboxylic acid + Amine"
    ),
    # Ether → two alcohols (Williamson retro)
    (
        "[OD2:1]([#6:2])[#6:3]",
        "[OX2H:1][#6:2]", "[#6:3][OH]",
        "Williamson ether → Alcohol + Alkyl halide"
    ),
    # Imine → aldehyde/ketone + amine
    (
        "[CX3:1]=[NX2:2]",
        "[CX3:1]=O", "[NX3H2:2]",
        "Imine → Carbonyl compound + Amine"
    ),
    # Alkene (Heck retro) → aryl halide + alkene
    (
        "[c:1][CH:2]=[CH2:3]",
        "[c:1][Br]", "[CH2:2]=[CH2:3]",
        "Heck coupling → Aryl halide + Terminal alkene"
    ),
    # Alcohol → alkene (retro-hydration)
    (
        "[CX4:1][OX2H:2]",
        "[CX3:1]=[CX3]", "H2O",
        "Alcohol → Alkene + Water (retro-hydration)"
    ),
    # Biaryl (Suzuki retro)
    (
        "[c:1][c:2]",
        "[c:1][Br]", "[c:2]B(O)O",
        "Suzuki coupling → Aryl halide + Boronic acid"
    ),
]

def retrosynthesis_html(product_smiles: str) -> str:
    """
    Heuristic retrosynthetic disconnection of a product SMILES.
    Applies SMARTS-based rules and returns an HTML panel showing
    possible precursor pairs.
    """
    if not product_smiles.strip():
        return "<i style='color:#888'>Enter a product SMILES to analyse retrosynthetically.</i>"

    mol = Chem.MolFromSmiles(product_smiles.strip())
    if mol is None:
        return "<i style='color:#e63946'>Invalid product SMILES.</i>"

    hits = []
    for prod_smarts, r1_smarts, r2_smarts, label in RETRO_RULES:
        patt = Chem.MolFromSmarts(prod_smarts)
        if patt and mol.HasSubstructMatch(patt):
            hits.append((label, r1_smarts, r2_smarts))

    if not hits:
        # Fallback: show functional group analysis as hints
        fg_list = detect_functional_groups(mol)
        fg_str  = ", ".join(fg_list) if fg_list else "none detected"
        return (
            f'<div style="padding:10px;background:#fff8e1;border:1px solid #ffe082;'
            f'border-radius:8px">'
            f'<p style="font-weight:600;color:#795548">⚗️ No automated disconnection found.</p>'
            f'<p style="font-size:13px;color:#555;margin-top:6px">'
            f'Detected functional groups: <b>{fg_str}</b>.<br>'
            f'Consider manual disconnection at these positions.</p></div>'
        )

    # Build product SVG
    product_svg = smiles_to_svg(product_smiles, 260, 170)
    product_smi_short = (product_smiles[:50] + "…") if len(product_smiles) > 50 else product_smiles

    cards = []
    for i, (label, r1_smarts, r2_smarts) in enumerate(hits[:5]):
        # Try to draw the precursor SMARTS as molecules
        def _smarts_or_smiles_svg(s, w=180, h=130):
            mol_try = Chem.MolFromSmiles(s)
            if mol_try:
                return smiles_to_svg(Chem.MolToSmiles(mol_try), w, h)
            mol_try2 = Chem.MolFromSmarts(s)
            if mol_try2:
                drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
                drawer.DrawMolecule(mol_try2)
                drawer.FinishDrawing()
                return drawer.GetDrawingText()
            return _placeholder_svg(s[:20], w, h)

        r1_svg = _smarts_or_smiles_svg(r1_smarts)
        r2_svg = _smarts_or_smiles_svg(r2_smarts)

        card = (
            f'<div style="border:1px solid #e0e0e0;border-radius:8px;padding:10px;'
            f'background:#fff;margin-bottom:10px">'
            f'<p style="font-size:12px;font-weight:600;color:#1565c0;margin-bottom:8px">'
            f'Route {i+1}: {label}</p>'
            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">'
            f'<div style="text-align:center">'
            f'<p style="font-size:10px;color:#888;margin-bottom:2px">Precursor A</p>'
            f'{r1_svg}'
            f'<p style="font-size:9px;color:#aaa;font-family:monospace">{r1_smarts[:30]}</p></div>'
            f'<div style="font-size:18px;color:#aaa">+</div>'
            f'<div style="text-align:center">'
            f'<p style="font-size:10px;color:#888;margin-bottom:2px">Precursor B</p>'
            f'{r2_svg}'
            f'<p style="font-size:9px;color:#aaa;font-family:monospace">{r2_smarts[:30]}</p></div>'
            f'</div></div>'
        )
        cards.append(card)

    html = (
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
        f'border-radius:10px;padding:14px">'
        f'<div style="text-align:center;margin-bottom:12px">'
        f'<p style="font-size:12px;font-weight:600;color:#444;margin-bottom:4px">🎯 Target molecule</p>'
        f'{product_svg}'
        f'<p style="font-size:10px;color:#999;font-family:monospace">{product_smi_short}</p></div>'
        f'<p style="font-size:13px;font-weight:600;color:#333;margin-bottom:8px">'
        f'⬆️ Retrosynthetic disconnections ({len(hits)} found):</p>'
        f'{"".join(cards)}'
        f'<p style="font-size:11px;color:#888;margin-top:8px">'
        f'⚠️ These are heuristic SMARTS-based disconnections, not ML predictions.</p></div>'
    )
    return html


# ─────────────────────────────────────────────────────────────────────────────
#  NEW FEATURE 3: 3D conformer viewer (py3Dmol via CDN in HTML iframe)
# ─────────────────────────────────────────────────────────────────────────────

def generate_3d_viewer_html(smiles: str) -> str:
    """
    Generate a self-contained HTML snippet with an embedded py3Dmol viewer
    showing the 3D conformer of the molecule. Uses UFF force-field minimisation.
    The viewer is embedded as an <iframe srcdoc=...>.
    """
    if not smiles or not smiles.strip():
        return "<i style='color:#888'>Enter a SMILES string to view 3D structure.</i>"

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return "<i style='color:#e63946'>Invalid SMILES — cannot generate 3D conformer.</i>"

    # Add hydrogens and generate 3D coords
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol_h, params)
        if result == -1:
            # Fallback to random coords
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
        mol_block = Chem.MolToMolBlock(mol_h)
    except Exception as e:
        return f"<i style='color:#e63946'>3D embedding failed: {e}</i>"

    # Escape mol block for embedding in JS string
    mol_block_escaped = mol_block.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    iframe_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ margin:0; padding:0; background:#1a1a2e; font-family:monospace; }}
  #viewer {{ width:100%; height:340px; position:relative; }}
  #controls {{
    position:absolute; top:8px; right:8px; z-index:100;
    display:flex; flex-direction:column; gap:4px;
  }}
  .ctrl-btn {{
    background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3);
    color:white; padding:4px 8px; font-size:11px; border-radius:4px;
    cursor:pointer; backdrop-filter:blur(4px);
  }}
  .ctrl-btn:hover {{ background:rgba(255,255,255,0.25); }}
  #info {{
    color:#aaa; font-size:11px; padding:4px 10px; background:#111;
    border-top:1px solid #333;
  }}
</style>
</head>
<body>
<div style="position:relative">
  <div id="viewer"></div>
  <div id="controls">
    <button class="ctrl-btn" onclick="setStyle('stick')">Stick</button>
    <button class="ctrl-btn" onclick="setStyle('sphere')">Sphere</button>
    <button class="ctrl-btn" onclick="setStyle('line')">Wire</button>
    <button class="ctrl-btn" onclick="viewer.zoomTo()">Reset</button>
  </div>
</div>
<div id="info">Drag to rotate · Scroll to zoom · Right-drag to pan</div>
<script>
var molblock = `{mol_block_escaped}`;
var viewer = $3Dmol.createViewer("viewer", {{
  backgroundColor: "0x1a1a2e",
}});
viewer.addModel(molblock, "sdf");
viewer.setStyle({{}}, {{stick: {{colorscheme: "Jmol", radius: 0.15}}}});
viewer.addSurface($3Dmol.SurfaceType.VDW, {{
  opacity: 0.08, color: "white"
}});
viewer.zoomTo();
viewer.render();

function setStyle(s) {{
  viewer.setStyle({{}}, {{}});
  if (s === "stick")   viewer.setStyle({{}}, {{stick: {{colorscheme:"Jmol", radius:0.15}}}});
  if (s === "sphere")  viewer.setStyle({{}}, {{sphere: {{colorscheme:"Jmol", scale:0.4}}}});
  if (s === "line")    viewer.setStyle({{}}, {{line: {{colorscheme:"Jmol"}}}});
  viewer.render();
}}
</script>
</body>
</html>"""

    # Encode for srcdoc
    import html as html_module
    srcdoc = html_module.escape(iframe_html)

    formula = rdMolDescriptors.CalcMolFormula(mol)
    n_atoms = mol.GetNumHeavyAtoms()
    n_atoms_h = mol_h.GetNumAtoms()

    return (
        f'<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden">'
        f'<div style="background:#1a1a2e;padding:8px 12px;display:flex;'
        f'justify-content:space-between;align-items:center">'
        f'<span style="color:#a0aec0;font-size:12px;font-family:monospace">'
        f'🧬 {formula} · {n_atoms} heavy atoms · {n_atoms_h} atoms with H</span>'
        f'<span style="color:#4a9eff;font-size:11px">UFF-minimised 3D</span></div>'
        f'<iframe srcdoc="{srcdoc}" width="100%" height="380" '
        f'frameborder="0" scrolling="no" '
        f'style="display:block;border:none"></iframe></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  NEW FEATURE 4: SMILES autocomplete suggestions
# ─────────────────────────────────────────────────────────────────────────────

COMMON_REAGENTS = [
    # Solvents
    ("Water",                   "O"),
    ("Methanol",                "CO"),
    ("Ethanol",                 "CCO"),
    ("Acetone",                 "CC(=O)C"),
    ("Dichloromethane (DCM)",   "ClCCl"),
    ("THF",                     "C1CCOC1"),
    ("DMF",                     "CN(C)C=O"),
    ("DMSO",                    "CS(=O)C"),
    ("Acetonitrile",            "CC#N"),
    ("Toluene",                 "Cc1ccccc1"),
    ("Hexane",                  "CCCCCC"),
    ("Ethyl acetate",           "CCOC(=O)C"),
    ("Diethyl ether",           "CCOCC"),
    # Common reagents
    ("Acetic acid",             "CC(=O)O"),
    ("Acetic anhydride",        "CC(=O)OC(=O)C"),
    ("Acetyl chloride",         "CC(=O)Cl"),
    ("Benzaldehyde",            "O=Cc1ccccc1"),
    ("Benzyl alcohol",          "OCc1ccccc1"),
    ("Bromobenzene",            "Brc1ccccc1"),
    ("n-Butyllithium",          "[Li]CCCC"),
    ("Chlorobenzene",           "Clc1ccccc1"),
    ("Dimethylamine",           "CNC"),
    ("EDC (coupling reagent)",  "CCN=C=NCCCN(C)C"),
    ("Ethylene glycol",         "OCCO"),
    ("Formaldehyde",            "C=O"),
    ("Grignard (EtMgBr)",       "CC[Mg]Br"),
    ("HOBt",                    "O=c1[nH]ncc2ccccc12"),
    ("Hydrogen peroxide",       "OO"),
    ("Iodobenzene",             "Ic1ccccc1"),
    ("LDA",                     "[Li]N(C(C)C)C(C)C"),
    ("Lithium aluminium hydride","[AlH4-].[Li+]"),
    ("mCPBA",                   "OOC(=O)c1cccc(Cl)c1"),
    ("Methylamine",             "CN"),
    ("NaBH4",                   "[BH4-].[Na+]"),
    ("NaOH",                    "[Na+].[OH-]"),
    ("Palladium (catalyst)",    "[Pd]"),
    ("Phenylboronic acid",      "OB(O)c1ccccc1"),
    ("Phosphorus tribromide",   "BrP(Br)Br"),
    ("Potassium carbonate",     "[K+].[K+].[O-]C([O-])=O"),
    ("Pyridine",                "c1ccncc1"),
    ("Sodium azide",            "[Na+].[N-]=[N+]=[N-]"),
    ("Sodium hydride",          "[NaH]"),
    ("Thionyl chloride",        "ClS(=O)Cl"),
    ("Triethylamine",           "CCN(CC)CC"),
    ("Trimethylamine",          "CN(C)C"),
    ("Triphenylphosphine",      "P(c1ccccc1)(c1ccccc1)c1ccccc1"),
]

def get_reagent_suggestions(query: str) -> list:
    """Return list of (name, smiles) tuples matching the query."""
    if not query or not query.strip():
        return COMMON_REAGENTS[:10]
    q = query.lower().strip()
    # Match by name or SMILES prefix
    results = [
        (name, smi) for name, smi in COMMON_REAGENTS
        if q in name.lower() or smi.lower().startswith(q) or q in smi.lower()
    ]
    return results[:12] if results else []

def reagent_suggestions_html(query: str) -> str:
    """Return an HTML panel of clickable reagent suggestions."""
    suggestions = get_reagent_suggestions(query)
    if not suggestions:
        return (
            f'<div style="padding:10px;background:#fff8e1;border:1px solid #ffe082;'
            f'border-radius:8px;font-size:12px;color:#795548">'
            f'No reagents matching "{query}". Try: "methanol", "amine", "Pd", "NaH"...</div>'
        )
    items = ""
    for name, smi in suggestions:
        svg = smiles_to_svg(smi, 100, 70)
        items += (
            f'<div style="border:1px solid #e5e7eb;border-radius:6px;padding:6px;'
            f'text-align:center;cursor:pointer;min-width:110px;background:#fff;'
            f'transition:box-shadow 0.2s" '
            f'onmouseover="this.style.boxShadow=\'0 2px 8px rgba(0,0,0,0.15)\'" '
            f'onmouseout="this.style.boxShadow=\'none\'">'
            f'{svg}'
            f'<p style="font-size:10px;font-weight:600;color:#333;margin:3px 0 1px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:105px">'
            f'{name}</p>'
            f'<p style="font-size:9px;color:#888;font-family:monospace;'
            f'word-break:break-all">{smi[:18]}{"…" if len(smi)>18 else ""}</p>'
            f'</div>'
        )
    return (
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
        f'border-radius:10px;padding:10px">'
        f'<p style="font-size:12px;color:#555;margin-bottom:8px;font-weight:600">'
        f'💊 {len(suggestions)} reagent(s) found — copy the SMILES to use above:</p>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px">{items}</div></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Confidence / beam helpers  (updated with threshold filter)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores(scores: list) -> list:
    if not scores:
        return []
    max_s = max(scores)
    exps  = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [round(100 * e / total, 1) for e in exps]


def format_candidates(candidates, scores=None, conf_threshold: float = 0.0) -> str:
    """Format beam candidates, hiding those below conf_threshold (0–100)."""
    if not candidates:
        return "No candidates returned."
    if scores is None:
        scores = list(range(len(candidates), 0, -1))
    pcts  = normalize_scores(scores)
    width = 22
    lines = [f"{'Rank':<6} {'Conf':>6}  {'Bar':<{width}}  SMILES"]
    lines.append("─" * 90)
    shown = 0
    for i, (smi, pct) in enumerate(zip(candidates, pcts), 1):
        if pct < conf_threshold:
            continue
        shown += 1
        filled = int(pct / 5)
        bar    = "█" * filled + "░" * (width - filled)
        lines.append(f"  #{i:<4} {pct:>5.1f}%  {bar}  {smi}")
    if shown == 0:
        lines.append(f"  (all candidates below {conf_threshold:.0f}% threshold — lower the slider)")
    elif shown < len(candidates):
        hidden = len(candidates) - shown
        lines.append(f"  … {hidden} candidate(s) hidden (below {conf_threshold:.0f}% threshold)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def char_accuracy(pred, actual):
    if not actual:
        return 0.0
    matches = sum(p == a for p, a in zip(pred, actual))
    return matches / max(len(pred), len(actual))

def levenshtein_accuracy(pred, actual):
    if not actual:
        return 0.0
    m, n = len(pred), len(actual)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if pred[i-1] == actual[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev  = temp
    return 1.0 - dp[n] / max(m, n)

def prefix_match(pred, actual):
    if not actual:
        return 0.0
    common = 0
    for p, a in zip(pred, actual):
        if p == a:
            common += 1
        else:
            break
    return common / max(len(pred), len(actual))


# ─────────────────────────────────────────────────────────────────────────────
#  History log
# ─────────────────────────────────────────────────────────────────────────────

history_log: list = []

def log_prediction(reactant, predicted, actual, model, bw,
                   char_acc=None, lev_acc=None, tanimoto=None, rxn_type=None):
    history_log.append({
        "time":      datetime.datetime.now().strftime("%H:%M:%S"),
        "reactant":  reactant,
        "predicted": predicted,
        "actual":    actual or "",
        "model":     model,
        "beam":      bw,
        "rxn_type":  rxn_type or "",
        "char_acc":  f"{100*char_acc:.1f}%" if char_acc  is not None else "—",
        "lev_acc":   f"{100*lev_acc:.1f}%"  if lev_acc   is not None else "—",
        "tanimoto":  f"{tanimoto:.3f}"       if tanimoto  is not None else "—",
        "starred":   False,
    })


def render_history(show_starred_only=False) -> str:
    entries = [h for h in history_log if h["starred"]] if show_starred_only else history_log
    if not entries:
        return "No predictions yet." if not show_starred_only else "No starred predictions."
    lines = [
        f"{'Time':^10} {'Model':^12} {'Beam':^5} {'Type':^22} "
        f"{'CharAcc':^8} {'LevAcc':^7} {'Tan':^6}  Reactant → Predicted"
    ]
    lines.append("─" * 130)
    for h in reversed(entries[-100:]):
        star = "⭐" if h["starred"] else "  "
        lines.append(
            f"{star}{h['time']:^10} {h['model']:^12} {str(h['beam']):^5} "
            f"{h['rxn_type'][:20]:^22} {h['char_acc']:^8} {h['lev_acc']:^7} "
            f"{h['tanimoto']:^6}  "
            f"{h['reactant'][:28]:<30} → {h['predicted'][:35]}"
        )
    return "\n".join(lines)


def star_last_prediction() -> str:
    if not history_log:
        return "No predictions to star."
    history_log[-1]["starred"] = True
    return f"⭐ Starred: {history_log[-1]['predicted'][:50]}"


# ─────────────────────────────────────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_history_csv() -> str:
    if not history_log:
        return None
    path = "/tmp/reaction_history.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history_log[0].keys()))
        w.writeheader()
        w.writerows(history_log)
    return path


def export_batch_csv(batch_rows: list) -> str:
    if not batch_rows:
        return None
    path = "/tmp/batch_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(batch_rows[0].keys()))
        w.writeheader()
        w.writerows(batch_rows)
    return path


def export_custom_csv(batch_rows: list) -> str:
    if not batch_rows:
        return None
    path = "/tmp/custom_batch_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(batch_rows[0].keys()))
        w.writeheader()
        w.writerows(batch_rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Predictor wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _get_predictor(model_choice):
    if model_choice == "GRU":
        return gru_predictor
    if model_choice == "Transformer":
        return tf_predictor
    return ensemble


def _call_predict_topk(predictor, reactant_smiles, beam_width, topk):
    sig    = inspect.signature(predictor.predict_topk)
    kwargs = {"beam_width": beam_width}
    if "topk" in sig.parameters:
        kwargs["topk"] = topk
    result = predictor.predict_topk(reactant_smiles, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        candidates, scores = result
    else:
        candidates = list(result)
        scores     = list(range(len(candidates), 0, -1))
    return candidates[:topk], scores[:topk]


def predict_topk_wrap(reactant_smiles, model_choice, beam_width, topk, conf_threshold=0.0):
    beam_width = int(beam_width)
    topk       = int(topk)
    predictor  = _get_predictor(model_choice)
    if hasattr(predictor, "predict_topk"):
        candidates, scores = _call_predict_topk(predictor, reactant_smiles, beam_width, topk)
    else:
        top1       = predictor.predict(reactant_smiles, beam_width=beam_width)
        candidates = [top1]
        scores     = [1.0]
    cand_str = format_candidates(candidates, scores, conf_threshold=conf_threshold)
    return candidates[0] if candidates else "", cand_str


def evaluate_reaction(reactant_smiles, actual_smiles, model_choice, beam_width, topk,
                      conf_threshold=0.0):
    beam_width = int(beam_width)
    topk       = int(topk)
    predictor  = _get_predictor(model_choice)

    if hasattr(predictor, "predict_topk"):
        candidates, scores = _call_predict_topk(predictor, reactant_smiles, beam_width, topk)
        predicted = candidates[0] if candidates else ""
    else:
        predicted  = predictor.predict(reactant_smiles, beam_width=beam_width)
        candidates = [predicted]
        scores     = [1.0]

    exact    = predicted == actual_smiles
    char_acc = char_accuracy(predicted, actual_smiles)
    lev_acc  = levenshtein_accuracy(predicted, actual_smiles)
    pre_acc  = prefix_match(predicted, actual_smiles)
    tan      = tanimoto_similarity(predicted, actual_smiles)
    rxn_type = classify_reaction(reactant_smiles)

    log_prediction(reactant_smiles, predicted, actual_smiles,
                   model_choice, beam_width, char_acc, lev_acc, tan, rxn_type)

    match_icon = "✅ Exact Match!" if exact else "❌ No Exact Match"
    tan_str    = f"{tan:.3f}" if tan >= 0 else "N/A"
    metrics = (
        f"🔬 Predicted : {predicted}\n"
        f"🧪 Actual    : {actual_smiles}\n\n"
        f"{match_icon}\n"
        f"📊 Char accuracy     : {100*char_acc:.2f}%\n"
        f"📊 Levenshtein acc   : {100*lev_acc:.2f}%\n"
        f"📊 Prefix match      : {100*pre_acc:.2f}%\n"
        f"🔗 Tanimoto similarity: {tan_str}\n"
        f"🏷️  Reaction type     : {rxn_type}"
    )
    cand_str = format_candidates(candidates, scores, conf_threshold=conf_threshold)
    return predicted, metrics, cand_str, rxn_type


# ─────────────────────────────────────────────────────────────────────────────
#  Model comparison  (run all three models on same input)
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_models(reactant_smiles, actual_smiles, beam_width, topk):
    beam_width = int(beam_width)
    topk       = int(topk)
    results    = []
    models     = ["GRU", "Transformer", "Ensemble"]
    for m in models:
        pred = _get_predictor(m).predict(reactant_smiles, beam_width=beam_width)
        exact = pred == actual_smiles if actual_smiles.strip() else None
        ca    = char_accuracy(pred, actual_smiles) if actual_smiles.strip() else None
        la    = levenshtein_accuracy(pred, actual_smiles) if actual_smiles.strip() else None
        tan   = tanimoto_similarity(pred, actual_smiles) if actual_smiles.strip() else None
        results.append((m, pred, exact, ca, la, tan))

    lines = [f"{'Model':<14} {'Char%':>6} {'Lev%':>6} {'Tan':>6} {'Match':>6}  Predicted SMILES"]
    lines.append("─" * 100)
    for m, pred, exact, ca, la, tan in results:
        ca_s    = f"{100*ca:.1f}%" if ca  is not None else "—"
        la_s    = f"{100*la:.1f}%" if la  is not None else "—"
        tan_s   = f"{tan:.3f}"     if tan is not None else "—"
        match_s = ("✓" if exact else "✗") if exact is not None else "—"
        lines.append(f"{m:<14} {ca_s:>6} {la_s:>6} {tan_s:>6} {match_s:>6}  {pred}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch helpers
# ─────────────────────────────────────────────────────────────────────────────

_batch_rows: list   = []
_custom_rows: list  = []


def run_batch_eval(n, model, bw, topk, conf_threshold=0):
    n    = int(n)
    bw   = int(bw)
    topk = int(topk)
    rows = []
    rxn_type_stats = {}
    lines = [
        f"{'#':<5} {'CharAcc':>7} {'LevAcc':>7} {'Tan':>6} {'Match':>6}  "
        f"Reactant  →  Predicted  (Actual)"
    ]
    lines.append("─" * 120)
    exact_count = char_total = lev_total = tan_total = 0

    for i, r in enumerate(test_reactions[:n]):
        predicted, _, _, rxn_type = evaluate_reaction(
            r["input"], r["output"], model, bw, topk
        )
        actual = r["output"]
        ca     = char_accuracy(predicted, actual)
        la     = levenshtein_accuracy(predicted, actual)
        tan    = tanimoto_similarity(predicted, actual)
        exact  = predicted == actual

        exact_count += int(exact)
        char_total  += ca
        lev_total   += la
        if tan >= 0:
            tan_total += tan

        rxn_type_stats.setdefault(rxn_type, {"n": 0, "exact": 0, "char": 0.0})
        rxn_type_stats[rxn_type]["n"]     += 1
        rxn_type_stats[rxn_type]["exact"] += int(exact)
        rxn_type_stats[rxn_type]["char"]  += ca

        match_str = "✓" if exact else "✗"
        lines.append(
            f"{i:<5} {100*ca:>6.1f}% {100*la:>6.1f}% {tan:>5.3f} {match_str:>6}  "
            f"{r['input'][:25]:<27} → {predicted[:25]:<27}  ({actual[:22]})"
        )
        rows.append({
            "index": i, "reactant": r["input"],
            "predicted": predicted, "actual": actual,
            "exact_match": exact, "rxn_type": rxn_type,
            "char_accuracy": round(100*ca, 2),
            "lev_accuracy":  round(100*la, 2),
            "tanimoto":      round(tan, 4),
            "model": model, "beam_width": bw,
        })

    lines.append("─" * 120)
    lines.append(
        f"Summary ({n} samples) — "
        f"Exact: {100*exact_count/n:.1f}%  |  "
        f"Avg char: {100*char_total/n:.1f}%  |  "
        f"Avg Lev: {100*lev_total/n:.1f}%  |  "
        f"Avg Tanimoto: {tan_total/n:.3f}"
    )
    lines.append("\n── Per reaction-type breakdown ──")
    lines.append(f"{'Type':<28} {'Count':>6} {'Exact%':>8} {'AvgChar%':>9}")
    lines.append("─" * 55)
    for rtype, st in sorted(rxn_type_stats.items(), key=lambda x: -x[1]["n"]):
        cnt = st["n"]
        lines.append(
            f"{rtype:<28} {cnt:>6} {100*st['exact']/cnt:>7.1f}% "
            f"{100*st['char']/cnt:>8.1f}%"
        )
    _batch_rows[:] = rows
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  NEW FEATURE 5: Confidence calibration plot (HTML canvas)
# ─────────────────────────────────────────────────────────────────────────────

def confidence_calibration_html(batch_rows: list) -> str:
    """
    Given batch_rows (list of dicts with 'char_accuracy' and optionally a
    mock confidence derived from tanimoto), produce an HTML scatter plot
    rendered on a <canvas> element showing calibration.
    """
    if not batch_rows:
        return (
            "<i style='color:#888'>Run a batch evaluation first, "
            "then click 'Show Calibration Plot'.</i>"
        )

    # Use char_accuracy as a proxy for "actual correctness"
    # Use tanimoto as a proxy for model confidence (higher = more confident)
    points = []
    for row in batch_rows:
        ca  = row.get("char_accuracy", 0)
        tan = row.get("tanimoto", 0)
        if isinstance(ca, (int, float)) and isinstance(tan, (int, float)) and tan >= 0:
            # Normalise tanimoto 0-1 → 0-100 for "confidence" axis
            points.append({"conf": round(tan * 100, 1), "acc": round(float(ca), 1)})

    if not points:
        return "<i style='color:#888'>No valid data points in batch results.</i>"

    import json
    points_json = json.dumps(points)

    # Compute calibration bins (deciles)
    bins = [{"lo": i*10, "hi": (i+1)*10, "confs": [], "accs": []} for i in range(10)]
    for p in points:
        b = min(int(p["conf"] // 10), 9)
        bins[b]["confs"].append(p["conf"])
        bins[b]["accs"].append(p["acc"])

    bin_data = []
    for b in bins:
        if b["confs"]:
            avg_conf = sum(b["confs"]) / len(b["confs"])
            avg_acc  = sum(b["accs"])  / len(b["accs"])
            n        = len(b["confs"])
            bin_data.append({"conf": round(avg_conf, 1), "acc": round(avg_acc, 1), "n": n})
    bin_json = json.dumps(bin_data)

    html = f"""
<div style="background:#1e2030;border-radius:10px;padding:16px;font-family:monospace">
  <p style="color:#a0aec0;font-size:13px;margin-bottom:12px;font-weight:600">
    📉 Confidence Calibration Plot
    <span style="font-size:10px;color:#718096;font-weight:400">
      (x = Tanimoto similarity as proxy confidence, y = char accuracy)
    </span>
  </p>
  <canvas id="calibCanvas" width="620" height="360"
    style="background:#151722;border-radius:8px;display:block;max-width:100%"></canvas>
  <div id="calibLegend" style="margin-top:8px;font-size:11px;color:#718096"></div>
</div>
<script>
(function() {{
  var points = {points_json};
  var bins   = {bin_json};
  var canvas = document.getElementById('calibCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var PAD = {{l:52, r:24, t:24, b:48}};
  var pw = W - PAD.l - PAD.r;
  var ph = H - PAD.t - PAD.b;

  function toX(v) {{ return PAD.l + v / 100 * pw; }}
  function toY(v) {{ return PAD.t + ph - v / 100 * ph; }}

  // Grid
  ctx.strokeStyle = '#2d3150';
  ctx.lineWidth = 1;
  for (var g = 0; g <= 10; g++) {{
    var gx = toX(g * 10), gy = toY(g * 10);
    ctx.beginPath(); ctx.moveTo(gx, PAD.t); ctx.lineTo(gx, PAD.t + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.l, gy); ctx.lineTo(PAD.l + pw, gy); ctx.stroke();
    ctx.fillStyle = '#4a5568'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText(g*10+'%', gx, H - PAD.b + 14);
    ctx.textAlign = 'right';
    ctx.fillText(g*10+'%', PAD.l - 6, gy + 4);
  }}

  // Perfect calibration line
  ctx.strokeStyle = '#4a9eff'; ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(toX(0), toY(0)); ctx.lineTo(toX(100), toY(100)); ctx.stroke();
  ctx.setLineDash([]);

  // Scatter points
  points.forEach(function(p) {{
    ctx.beginPath();
    ctx.arc(toX(p.conf), toY(p.acc), 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(100,200,150,0.4)';
    ctx.fill();
  }});

  // Calibration bins as bar chart overlay
  bins.forEach(function(b) {{
    var x = toX(b.conf), y = toY(b.acc);
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#f6ad55'; ctx.fill();
    ctx.strokeStyle = '#1e2030'; ctx.lineWidth = 1.5; ctx.stroke();
    // label n
    ctx.fillStyle = '#a0aec0'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
    ctx.fillText('n='+b.n, x, y - 10);
  }});

  // Axes labels
  ctx.fillStyle = '#718096'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
  ctx.fillText('Tanimoto confidence proxy (%)', W/2, H - 4);
  ctx.save(); ctx.translate(14, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Char accuracy (%)', 0, 0); ctx.restore();

  // Legend
  var leg = document.getElementById('calibLegend');
  if (leg) {{
    leg.innerHTML =
      '<span style="color:#64c896">● scatter ({n} pts)</span>  '.replace('{{n}}', points.length) +
      '<span style="color:#f6ad55">● bin avg (deciles)</span>  ' +
      '<span style="color:#4a9eff">--- perfect calibration</span>';
  }}
}})();
</script>
""".replace("{n}", str(len(points)))

    return html


# ─────────────────────────────────────────────────────────────────────────────
#  NEW FEATURE 6: Error analysis tab
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis_text(batch_rows: list, min_lev_error: float = 0.0) -> str:
    """
    Filter batch_rows to failures (exact_match == False),
    sort by Levenshtein distance (worst first), and return a formatted table.
    min_lev_error: minimum Levenshtein *error* (1 - lev_acc / 100) to include.
    """
    if not batch_rows:
        return "Run a batch evaluation first to see error analysis."

    failures = []
    for row in batch_rows:
        if row.get("exact_match") in (True, "True"):
            continue
        lev_acc = row.get("lev_accuracy", 0)
        if isinstance(lev_acc, str):
            try:
                lev_acc = float(lev_acc)
            except ValueError:
                lev_acc = 0.0
        lev_err = 100.0 - float(lev_acc)   # how bad (0 = perfect, 100 = totally wrong)
        if lev_err < min_lev_error:
            continue
        failures.append({**row, "_lev_err": lev_err})

    if not failures:
        return (
            f"✅ No failures found with Levenshtein error ≥ {min_lev_error:.0f}%!\n"
            "(Either all exact matches, or lower the threshold.)"
        )

    failures.sort(key=lambda x: x["_lev_err"], reverse=True)

    lines = [
        f"{'#':<5} {'LevErr':>7} {'Tan':>6} {'Type':^22}  Reactant → Predicted  (Actual)",
        "─" * 130,
    ]
    for i, row in enumerate(failures[:50], 1):
        tan = row.get("tanimoto", "—")
        try:
            tan = f"{float(tan):.3f}"
        except (ValueError, TypeError):
            tan = "—"
        rxn_type = str(row.get("rxn_type", ""))[:20]
        reactant  = str(row.get("reactant",  ""))[:22]
        predicted = str(row.get("predicted", ""))[:22]
        actual    = str(row.get("actual",    ""))[:20]
        lines.append(
            f"{i:<5} {row['_lev_err']:>6.1f}%  {tan:>6}  {rxn_type:^22}  "
            f"{reactant:<24} → {predicted:<24}  ({actual})"
        )

    lines.append("─" * 130)
    lines.append(
        f"Total failures shown: {len(failures)}  |  "
        f"Avg Lev error: {sum(r['_lev_err'] for r in failures)/len(failures):.1f}%  |  "
        f"Worst: {failures[0]['_lev_err']:.1f}%"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Upload CSV batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_custom_batch(csv_file, model, bw, topk):
    if csv_file is None:
        return "No file uploaded.", None
    bw   = int(bw)
    topk = int(topk)
    rows = []
    lines = [f"{'#':<5} {'CharAcc':>7} {'LevAcc':>7} {'Tan':>6} {'Match':>6}  Reactant → Predicted"]
    lines.append("─" * 100)
    exact_count = char_total = lev_total = 0
    n = 0

    try:
        with open(csv_file.name, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                reactant = row.get("input", row.get("reactant", row.get("smiles", ""))).strip()
                actual   = row.get("output", row.get("product", row.get("actual", ""))).strip()
                if not reactant:
                    continue
                predicted, _, _, rxn_type = evaluate_reaction(reactant, actual, model, bw, topk)
                ca    = char_accuracy(predicted, actual) if actual else None
                la    = levenshtein_accuracy(predicted, actual) if actual else None
                tan   = tanimoto_similarity(predicted, actual) if actual else None
                exact = (predicted == actual) if actual else None

                if ca is not None:
                    exact_count += int(exact)
                    char_total  += ca
                    lev_total   += la
                    n += 1

                lines.append(
                    f"{i:<5} "
                    f"{f'{100*ca:.1f}%' if ca is not None else '—':>7} "
                    f"{f'{100*la:.1f}%' if la is not None else '—':>7} "
                    f"{f'{tan:.3f}' if tan is not None else '—':>6} "
                    f"{'✓' if exact else ('✗' if exact is False else '—'):>6}  "
                    f"{reactant[:25]:<27} → {predicted[:30]}"
                )
                rows.append({
                    "index": i, "reactant": reactant,
                    "predicted": predicted, "actual": actual,
                    "exact_match": str(exact), "rxn_type": rxn_type,
                    "char_accuracy": round(100*ca, 2) if ca is not None else "",
                    "lev_accuracy":  round(100*la, 2) if la is not None else "",
                    "tanimoto":      round(tan, 4)    if tan is not None else "",
                    "model": model, "beam_width": bw,
                })
    except Exception as e:
        return f"Error reading CSV: {e}", None

    if n > 0:
        lines.append("─" * 100)
        lines.append(
            f"Summary ({n} evaluated) — "
            f"Exact: {100*exact_count/n:.1f}%  |  "
            f"Avg char: {100*char_total/n:.1f}%  |  "
            f"Avg Lev: {100*lev_total/n:.1f}%"
        )
    _custom_rows[:] = rows
    out_path = export_custom_csv(rows)
    return "\n".join(lines), out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Chemical Reaction Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🧪 Chemical Reaction Predictor\n"
        "Predict product SMILES from reactant SMILES — **GRU · Transformer · Ensemble** — "
        "with molecular visualization, confidence scores, Tanimoto similarity, "
        "reaction type classification, atom mapping, retrosynthesis, 3D viewer, and more."
    )

    # ── Global controls ───────────────────────────────────────────────────────
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["GRU", "Transformer", "Ensemble"],
            value="Ensemble", label="Model"
        )
        beam_width = gr.Slider(
            minimum=1, maximum=10, value=3, step=1, label="Beam width"
        )
        topk_choice = gr.Slider(
            minimum=1, maximum=10, value=5, step=1, label="Top-k candidates"
        )
        # NEW: Confidence threshold slider (now wired to filter)
        conf_threshold = gr.Slider(
            minimum=0, maximum=50, value=0, step=1,
            label="Min. confidence threshold (%) — hide candidates below this"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 1 — Manual input
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("Manual input"):
        gr.Markdown("Enter SMILES strings. Supplying the actual product enables match metrics.")

        with gr.Row():
            reactant_input = gr.Textbox(
                label="Reactant SMILES",
                placeholder="e.g. CCO.CC(=O)Cl",
                lines=2
            )
            actual_input = gr.Textbox(
                label="Actual product SMILES (optional)",
                placeholder="Leave blank for prediction only",
                lines=2
            )

        with gr.Row():
            predict_btn  = gr.Button("Predict reaction", variant="primary")
            validate_btn = gr.Button("Validate SMILES")
            canon_btn    = gr.Button("Canonicalize SMILES")
            star_btn     = gr.Button("⭐ Star last result")

        mol_display = gr.HTML(label="Molecular structures")

        predicted_out = gr.Textbox(
            label="Top-1 predicted product",
            interactive=False,
            lines=3,
            max_lines=10,
            show_copy_button=True,
        )

        candidates_out = gr.Textbox(
            label="Beam candidates + confidence (filtered by threshold)",
            interactive=False,
            lines=15,
            max_lines=50,
            show_copy_button=True,
        )

        metrics_out  = gr.Textbox(label="Match metrics", lines=9, interactive=False)
        diff_display = gr.HTML(label="Character-level diff (predicted vs actual)")

        rxn_smarts_out = gr.Textbox(
            label="Reaction SMARTS (reactant>>product)",
            interactive=False, lines=2, show_copy_button=True
        )

        validate_out = gr.Textbox(
            label="SMILES analysis / validation",
            lines=14, interactive=False
        )

        star_msg = gr.Textbox(label="", interactive=False, lines=1)

        # ── Handlers ──────────────────────────────────────────────────────────
        def on_predict(reactant, actual, model, bw, topk, conf_thr):
            reactant = reactant.strip()
            actual   = actual.strip()
            if actual:
                predicted, metrics, cand_str, rxn_type = evaluate_reaction(
                    reactant, actual, model, bw, topk, conf_threshold=conf_thr
                )
                diff    = smiles_diff_html(predicted, actual)
                rx_sma  = reaction_smarts(reactant, predicted)
            else:
                predicted, cand_str = predict_topk_wrap(
                    reactant, model, bw, topk, conf_threshold=conf_thr
                )
                rxn_type = classify_reaction(reactant)
                log_prediction(reactant, predicted, "", model, bw, rxn_type=rxn_type)
                metrics  = (
                    f"🏷️  Reaction type: {rxn_type}\n"
                    "Provide actual SMILES to see match metrics."
                )
                diff    = "<i style='color:#888'>Supply actual SMILES to see diff.</i>"
                rx_sma  = reaction_smarts(reactant, predicted)
            mol_html = side_by_side_html(reactant, predicted, actual, rxn_type)
            return mol_html, predicted, cand_str, metrics, diff, rx_sma

        predict_btn.click(
            fn=on_predict,
            inputs=[reactant_input, actual_input, model_choice, beam_width,
                    topk_choice, conf_threshold],
            outputs=[mol_display, predicted_out, candidates_out,
                     metrics_out, diff_display, rxn_smarts_out]
        )

        # Re-filter candidates when threshold slider changes (without re-predicting)
        def on_threshold_change(reactant, actual, model, bw, topk, conf_thr):
            """Re-run predict to update the candidate list with the new threshold."""
            return on_predict(reactant, actual, model, bw, topk, conf_thr)

        conf_threshold.change(
            fn=on_threshold_change,
            inputs=[reactant_input, actual_input, model_choice, beam_width,
                    topk_choice, conf_threshold],
            outputs=[mol_display, predicted_out, candidates_out,
                     metrics_out, diff_display, rxn_smarts_out]
        )

        def on_validate(reactant, actual):
            parts = []
            if reactant.strip():
                parts.append("── Reactant ──\n" + validate_smiles(reactant))
            if actual.strip():
                parts.append("\n── Actual product ──\n" + validate_smiles(actual))
            return "\n".join(parts) if parts else "Enter at least one SMILES to validate."

        validate_btn.click(
            fn=on_validate,
            inputs=[reactant_input, actual_input],
            outputs=[validate_out]
        )

        def on_canonicalize(reactant, actual):
            return (
                canonicalize(reactant) if reactant.strip() else reactant,
                canonicalize(actual)   if actual.strip()   else actual,
            )

        canon_btn.click(
            fn=on_canonicalize,
            inputs=[reactant_input, actual_input],
            outputs=[reactant_input, actual_input]
        )

        star_btn.click(fn=star_last_prediction, inputs=[], outputs=[star_msg])

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 2 — Test set browser
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("Test set browser"):
        gr.Markdown(f"Browse predictions on the test set ({len(test_reactions)} samples).")

        sample_idx = gr.Number(
            value=0, label=f"Sample index (0 – {len(test_reactions)-1})", precision=0
        )
        run_btn = gr.Button("Run", variant="primary")

        mol_display_test = gr.HTML(label="Molecular structures")

        with gr.Row():
            reactant_disp  = gr.Textbox(
                label="Reactant SMILES", interactive=False,
                lines=2, show_copy_button=True
            )
            predicted_disp = gr.Textbox(
                label="Predicted product", interactive=False,
                lines=2, show_copy_button=True
            )

        metrics_disp = gr.Textbox(
            label="Match metrics", interactive=False, lines=9
        )
        candidates_disp = gr.Textbox(
            label="Beam candidates", interactive=False,
            lines=15, max_lines=50
        )
        diff_disp_test = gr.HTML(label="Character-level diff")

        def run_on_test_sample(idx, model, bw, topk, conf_thr):
            idx = int(idx)
            if idx < 0 or idx >= len(test_reactions):
                msg = f"Index out of range (0–{len(test_reactions)-1})."
                return "", "", "", msg, "", ""
            r = test_reactions[idx]
            predicted, metrics, cand_str, rxn_type = evaluate_reaction(
                r["input"], r["output"], model, bw, topk, conf_threshold=conf_thr
            )
            mol_html = side_by_side_html(r["input"], predicted, r["output"], rxn_type)
            diff     = smiles_diff_html(predicted, r["output"])
            return mol_html, r["input"], predicted, metrics, cand_str, diff

        run_btn.click(
            fn=run_on_test_sample,
            inputs=[sample_idx, model_choice, beam_width, topk_choice, conf_threshold],
            outputs=[mol_display_test, reactant_disp, predicted_disp,
                     metrics_disp, candidates_disp, diff_disp_test]
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 3 — Batch evaluate (built-in test set)
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("Batch evaluate"):
        gr.Markdown(
            "Run over a batch from the built-in test set. "
            "Results include per-reaction-type breakdown and Tanimoto."
        )

        batch_size = gr.Slider(minimum=5, maximum=200, value=20, step=5,
                               label="Number of samples")

        with gr.Row():
            batch_btn  = gr.Button("Run batch", variant="primary")
            export_btn = gr.Button("Export to CSV")

        batch_out    = gr.Textbox(label="Batch results", lines=30,
                                  interactive=False, max_lines=200)
        csv_download = gr.File(label="Download CSV", interactive=False)

        batch_btn.click(
            fn=run_batch_eval,
            inputs=[batch_size, model_choice, beam_width, topk_choice],
            outputs=[batch_out]
        )
        export_btn.click(
            fn=lambda: export_batch_csv(_batch_rows),
            inputs=[], outputs=[csv_download]
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 4 — Upload your own CSV
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("Upload CSV"):
        gr.Markdown(
            "Upload a CSV with an `input` (reactant SMILES) column and optionally "
            "an `output` (actual product) column. Predictions run on every row."
        )
        csv_upload   = gr.File(label="Upload CSV", file_types=[".csv"])
        custom_run   = gr.Button("Run predictions", variant="primary")
        custom_out   = gr.Textbox(label="Results", lines=25,
                                  interactive=False, max_lines=200)
        custom_dl    = gr.File(label="Download results CSV", interactive=False)

        custom_run.click(
            fn=run_custom_batch,
            inputs=[csv_upload, model_choice, beam_width, topk_choice],
            outputs=[custom_out, custom_dl]
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 5 — Model comparison
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("Model comparison"):
        gr.Markdown(
            "Run the same input through **all three models** simultaneously "
            "and compare their predictions side by side."
        )
        with gr.Row():
            cmp_reactant = gr.Textbox(
                label="Reactant SMILES", placeholder="e.g. CCO.CC(=O)Cl", lines=2
            )
            cmp_actual   = gr.Textbox(
                label="Actual product SMILES (optional)", lines=2
            )
        cmp_btn  = gr.Button("Compare all models", variant="primary")
        cmp_out  = gr.Textbox(label="Comparison table",
                              lines=10, interactive=False, max_lines=20)
        cmp_mols = gr.HTML(label="GRU  |  Transformer  |  Ensemble predictions")

        def on_compare(reactant, actual, bw, topk):
            reactant = reactant.strip()
            actual   = actual.strip()
            table    = compare_all_models(reactant, actual, bw, topk)
            html_parts = ['<div style="display:flex;flex-wrap:wrap;gap:12px">']
            for m in ["GRU", "Transformer", "Ensemble"]:
                pred = _get_predictor(m).predict(reactant, beam_width=int(bw))
                html_parts.append(
                    f'<div style="border:1px solid #ddd;border-radius:8px;padding:8px">'
                    f'{smiles_html(m, pred, 220, 160)}</div>'
                )
            html_parts.append("</div>")
            return table, "".join(html_parts)

        cmp_btn.click(
            fn=on_compare,
            inputs=[cmp_reactant, cmp_actual, beam_width, topk_choice],
            outputs=[cmp_out, cmp_mols]
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 6 — History log
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("History log"):
        gr.Markdown("All predictions made this session. ⭐ = starred via Manual input tab.")

        with gr.Row():
            show_starred   = gr.Checkbox(label="Show starred only", value=False)
            refresh_btn    = gr.Button("Refresh", variant="secondary")
            export_hist_btn = gr.Button("Export to CSV")

        history_out  = gr.Textbox(label="Prediction history",
                                  lines=25, interactive=False, max_lines=200)
        hist_csv_dl  = gr.File(label="Download history CSV", interactive=False)

        refresh_btn.click(
            fn=render_history,
            inputs=[show_starred],
            outputs=[history_out]
        )
        export_hist_btn.click(fn=export_history_csv, inputs=[], outputs=[hist_csv_dl])

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 7 — SMILES toolkit
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("SMILES toolkit"):
        gr.Markdown("Standalone molecule drawing, property analysis, and validation.")

        toolkit_input = gr.Textbox(
            label="SMILES", placeholder="Enter any SMILES string", lines=2
        )
        with gr.Row():
            tk_mol_btn  = gr.Button("Draw molecule")
            tk_stat_btn = gr.Button("Analyze properties")
            tk_val_btn  = gr.Button("Validate")

        tk_mol_html  = gr.HTML(label="Structure")
        tk_stats_out = gr.Textbox(
            label="Molecular properties", lines=14, interactive=False
        )

        gr.Markdown("---\n### Tanimoto similarity between two molecules")
        with gr.Row():
            tan_smi1 = gr.Textbox(label="SMILES 1", lines=1)
            tan_smi2 = gr.Textbox(label="SMILES 2", lines=1)
        tan_btn = gr.Button("Compute Tanimoto")
        tan_out = gr.Textbox(label="Result", lines=2, interactive=False)

        def tk_draw(smi):
            if not smi.strip():
                return "<i>No SMILES entered.</i>"
            svg   = smiles_to_svg(smi, 420, 260)
            canon = canonicalize(smi) if Chem.MolFromSmiles(smi) else smi
            return (
                f'<div style="text-align:center">'
                f'<p style="font-size:11px;color:#888;font-family:monospace;'
                f'margin-bottom:6px;word-break:break-all">{canon}</p>'
                f'{svg}</div>'
            )

        tk_mol_btn.click(fn=tk_draw, inputs=[toolkit_input], outputs=[tk_mol_html])
        tk_stat_btn.click(fn=get_mol_properties,  inputs=[toolkit_input], outputs=[tk_stats_out])
        tk_val_btn.click(fn=validate_smiles,       inputs=[toolkit_input], outputs=[tk_stats_out])

        def on_tanimoto(s1, s2):
            t = tanimoto_similarity(s1.strip(), s2.strip())
            if t < 0:
                return "Could not compute — one or both SMILES are invalid."
            return (
                f"Tanimoto (Morgan r=2, 2048 bits): {t:.4f}\n"
                f"Similarity: {'very high (>0.85)' if t>0.85 else 'high (>0.7)' if t>0.7 else 'moderate (>0.4)' if t>0.4 else 'low'}"
            )

        tan_btn.click(fn=on_tanimoto, inputs=[tan_smi1, tan_smi2], outputs=[tan_out])

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 8 — NEW: Atom mapping
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("⚛️ Atom mapping"):
        gr.Markdown(
            "### Atom Mapping\n"
            "Visualise which atoms in the **reactant** correspond to atoms in the **product** "
            "using Maximum Common Substructure (MCS) matching. "
            "Matching atoms are shown in the same colour."
        )
        with gr.Row():
            am_reactant = gr.Textbox(
                label="Reactant SMILES",
                placeholder="e.g. CCO",
                lines=2
            )
            am_product  = gr.Textbox(
                label="Product SMILES",
                placeholder="e.g. CCOC(=O)C",
                lines=2
            )
        am_btn      = gr.Button("Generate atom mapping", variant="primary")
        am_html_out = gr.HTML(label="Atom-mapped structures")

        # Auto-fill from manual input tab
        gr.Markdown(
            "💡 **Tip:** After running a prediction in the **Manual input** tab, "
            "paste the predicted product SMILES here to see atom mapping."
        )

        am_btn.click(
            fn=generate_atom_mapping_html,
            inputs=[am_reactant, am_product],
            outputs=[am_html_out]
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 9 — NEW: Retrosynthesis mode
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("🔄 Retrosynthesis"):
        gr.Markdown(
            "### Retrosynthetic Analysis\n"
            "Enter a **target molecule** (product SMILES) to find possible "
            "precursor pairs using SMARTS-based disconnection rules.\n\n"
            "Supports: ester hydrolysis, amide bond, Williamson ether, "
            "imine, Heck, Suzuki, retro-hydration."
        )
        retro_input = gr.Textbox(
            label="Target molecule (product SMILES)",
            placeholder="e.g. CCOC(=O)c1ccccc1  (ethyl benzoate)",
            lines=2
        )
        retro_btn = gr.Button("Analyse retrosynthesis", variant="primary")
        retro_html_out = gr.HTML(label="Retrosynthetic disconnections")

        retro_btn.click(
            fn=retrosynthesis_html,
            inputs=[retro_input],
            outputs=[retro_html_out]
        )

        gr.Markdown(
            "⚠️ *These are heuristic SMARTS disconnections for common reaction classes, "
            "not deep-learning retrosynthesis. For advanced retrosynthesis, "
            "consider tools like AiZynthFinder or ASKCOS.*"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 10 — NEW: 3D conformer viewer
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("🧬 3D Viewer"):
        gr.Markdown(
            "### 3D Conformer Viewer\n"
            "Generate and visualise a 3D conformer of any molecule using "
            "RDKit ETKDGv3 embedding + UFF force-field minimisation, "
            "displayed interactively via **3Dmol.js**.\n\n"
            "Drag to rotate · Scroll to zoom · Right-drag to pan"
        )
        viewer_input = gr.Textbox(
            label="SMILES",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O  (aspirin)",
            lines=2
        )
        viewer_btn = gr.Button("Generate 3D view", variant="primary")
        viewer_html = gr.HTML(label="3D structure (interactive)")

        gr.Markdown(
            "💡 Style buttons: **Stick** (default) · **Sphere** (space-filling) · **Wire** (wireframe)"
        )

        viewer_btn.click(
            fn=generate_3d_viewer_html,
            inputs=[viewer_input],
            outputs=[viewer_html]
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 11 — NEW: Reagent autocomplete / browser
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("💊 Reagent browser"):
        gr.Markdown(
            "### Reagent & Solvent Browser\n"
            "Search common reagents, solvents, and coupling agents by name or SMILES fragment. "
            "Copy the SMILES string into the **Manual input** tab to use as a reactant."
        )
        reagent_query = gr.Textbox(
            label="Search (name or SMILES fragment)",
            placeholder="e.g. amine, Pd, NaH, ester, chloride...",
            lines=1
        )
        with gr.Row():
            reagent_search_btn = gr.Button("Search", variant="primary")
            reagent_all_btn    = gr.Button("Show all common reagents")

        reagent_html_out = gr.HTML(label="Matching reagents")

        reagent_search_btn.click(
            fn=reagent_suggestions_html,
            inputs=[reagent_query],
            outputs=[reagent_html_out]
        )
        reagent_all_btn.click(
            fn=lambda: reagent_suggestions_html(""),
            inputs=[],
            outputs=[reagent_html_out]
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 12 — NEW: Confidence calibration plot
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("📉 Calibration plot"):
        gr.Markdown(
            "### Confidence Calibration Plot\n"
            "After running a **Batch evaluate**, click below to see how well "
            "the model's structural similarity (Tanimoto, used as a confidence proxy) "
            "correlates with character-level accuracy.\n\n"
            "A well-calibrated model's points should cluster near the diagonal."
        )
        calib_btn  = gr.Button("Show calibration plot (from last batch)", variant="primary")
        calib_html = gr.HTML(label="Calibration scatter plot")

        calib_btn.click(
            fn=lambda: confidence_calibration_html(_batch_rows),
            inputs=[],
            outputs=[calib_html]
        )

        gr.Markdown(
            "- **Green dots** = individual predictions\n"
            "- **Orange dots** = decile bin averages\n"
            "- **Blue dashed** = perfect calibration line\n\n"
            "*Note: Tanimoto similarity is used as a proxy for confidence since "
            "raw model log-probabilities are not exposed through the current predictor API.*"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Tab 13 — NEW: Error analysis
    # ═════════════════════════════════════════════════════════════════════════
    with gr.Tab("🔍 Error analysis"):
        gr.Markdown(
            "### Error Analysis\n"
            "After running a **Batch evaluate**, filter results to show only "
            "**failures** (non-exact-matches), sorted by Levenshtein error (worst first).\n\n"
            "Use the threshold slider to focus on the most severe errors."
        )
        err_threshold = gr.Slider(
            minimum=0, maximum=100, value=0, step=5,
            label="Minimum Levenshtein error % to include (0 = all failures)"
        )
        err_btn = gr.Button("Show error analysis (from last batch)", variant="primary")
        err_out = gr.Textbox(
            label="Error analysis — failures sorted by Levenshtein error",
            lines=30, interactive=False, max_lines=200
        )

        err_btn.click(
            fn=lambda thr: error_analysis_text(_batch_rows, min_lev_error=thr),
            inputs=[err_threshold],
            outputs=[err_out]
        )

        gr.Markdown(
            "💡 **LevErr** = 100% − Levenshtein accuracy. "
            "Higher = more different from actual. "
            "**Tan** = Tanimoto similarity (structural closeness)."
        )


demo.launch(share=True, debug=True)