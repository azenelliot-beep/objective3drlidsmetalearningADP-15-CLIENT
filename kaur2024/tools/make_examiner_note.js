const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
        AlignmentType, LevelFormat, BorderStyle, HeadingLevel } = require("docx");

const FONT = "Arial";
const W = 9638; // A4 text width with 2 cm margins (DXA)
const border = { style: BorderStyle.SINGLE, size: 4, color: "B0B0B0" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, opts = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!opts.bold, size: opts.size || 18, font: FONT })] })],
  });
}
function table(cols, header, rows, highlight = [], leftAll = false) {
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: cols,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) =>
        cell(h, cols[i], { bold: true, fill: "DCE6F1", align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) }),
      ...rows.map((r, k) => new TableRow({ children: r.map((v, i) =>
        cell(v, cols[i], { bold: highlight.includes(k), fill: highlight.includes(k) ? "FFF5D6" : undefined,
                           align: (i && !leftAll) ? AlignmentType.CENTER : AlignmentType.LEFT })) })),
    ],
  });
}
const p = (text, opts = {}) => new Paragraph({ spacing: { after: opts.after ?? 100 },
  children: [new TextRun({ text, size: opts.size || 20, font: FONT, bold: opts.bold, italics: opts.italics })] });
const runs = (parts, after = 60) => new Paragraph({ spacing: { after },
  children: parts.map(([t, b]) => new TextRun({ text: t, bold: !!b, size: 20, font: FONT })) });
const bullet = (parts) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 50 },
  children: (typeof parts === "string" ? [[parts]] : parts).map(([t, b]) => new TextRun({ text: t, bold: !!b, size: 20, font: FONT })) });
const h = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });


const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 30, bold: true },
        paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 1000, bottom: 1000, left: 1134, right: 1134 } } },
    children: [
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Kaur (2024) on TON_IoT: published work vs my reimplementation")] }),
      p("A. Kaur, “Intrusion Detection Approach for IIoT Traffic using Deep Recurrent Reinforcement Learning Assisted "
        + "Federated Learning,” IEEE Trans. Artificial Intelligence, 2024, DOI 10.1109/TAI.2024.3443787.",
        { italics: true, size: 18, after: 120 }),

      h("1. Their TON_IoT work (as published)"),
      bullet([["Model: ", true], ["GRU classifier (2 × 128, dropout 0.3) trained by federated learning; a DQN selects which clients take part each round."]]),
      bullet([["Reported on TON_IoT (Table V): ", true], ["99.95% accuracy, macro F1 99.99%, AUC 0.999 (Fig. 7)."]]),
      bullet([["Not stated for TON_IoT: ", true], ["IID or non-IID, number of clients, and the round of the result. Their 15-client non-IID experiment (Table VIII) is on a different dataset, Edge-IIoT."]]),

      h("2. My reimplementation on TON_IoT"),
      bullet([["Why: ", true], ["the paper gives no TON_IoT result under non-IID data, so I reran their method under the same conditions as my own model."]]),
      bullet([["Conditions: ", true], ["TON_IoT, non-IID (Dirichlet α = 0.5), 15 clients (DQN selects 8 per round), 5 local epochs, batch 20, fixed train/test split (test = 89,518 rows, identical to their Table II)."]]),
      bullet([["Use: ", true], ["my model is compared against this reimplementation under identical conditions (same data split, clients and 50-round budget)."]]),

      h("3. Results: claimed vs reimplemented"),
      table([2500, 1830, 900, 1050, 1050, 750, 778, 780],
        ["Source", "Conditions", "Rounds", "Accuracy", "Precision", "Recall", "F1", "AUC"],
        [
          ["Claimed in the paper (Table V, Fig. 7)", "TON_IoT; IID/non-IID and clients not stated", "not stated", "99.95", "98.63", "99.98", "99.99", "99.9"],
          ["Reimplemented", "TON_IoT, non-IID, 15 clients", "50", "76.69", "66.42", "45.86", "41.53", "95.41"],
          ["Reimplemented", "TON_IoT, non-IID, 15 clients", "100", "76.72", "64.14", "45.99", "41.91", "95.75"],
        ], [1, 2]),
      p("All values in %. Precision, recall and F1 are macro averages over the 10 classes.", { size: 16, italics: true, after: 100 }),

      h("4. How the reimplementation differs from the published work"),
      table([2000, 2400, 2500, 2738], ["Item", "Paper", "Reimplementation", "Reason"], [
        ["Data distribution", "Not stated for TON_IoT", "Non-IID, Dirichlet α = 0.5", "Match my model’s setting"],
        ["Clients", "Not stated (15 only on Edge-IIoT)", "15, DQN selects 8 per round", "Follows their Table VIII protocol"],
        ["Rounds", "100 (Table I)", "100; reported at 50 and 100", "50 = same budget as my model"],
        ["Train / test rows", "395,467 / 89,518 (Table II)", "371,485 / 89,518", "The file has too few rows for Table II’s train counts"],
        ["Optimiser", "Adam, lr 0.2", "SGD, lr 0.2", "Adam at 0.2 predicted one class only; SGD matches their Eqs. 17, 24"],
        ["Loss", "Eq. 20 (squared error)", "Softmax cross-entropy", "Eq. 20 does not fit 10-class classification"],
        ["GRU input", "Not stated", "Each record = sequence of length 1", "Paper does not define sequences"],
        ["Removed features", "“Unwanted features”", "Timestamp, IPs, ports, label, dns_query, http_uri", "Identifiers, and text columns with > 50 values"],
        ["DQN details", "Not stated", "Top 8 of 15; ε-greedy; 2 × 64 network", "Needed for the code to run"],
        ["Energy constants", "Partly given", "Added missing constants (e.g. CPU capacitance)", "Eqs. 1–3 incomplete as written"],
      ], [], true),

      h("5. Why the results differ"),
      bullet([["Scaling: ", true], ["min-max normalisation (their Eq. 19) squeezes byte, packet and duration values to ≈ 0, so the model relies on categorical flags; five attack classes are mostly missed."]]),
      bullet([["Optimiser: ", true], ["the published setting (Adam, lr 0.2) does not train; SGD was required."]]),
      bullet([["The paper’s own Table V is inconsistent: ", true], ["its per-class values imply 96.47% accuracy and 95.02% F1, not the reported 99.95% / 99.99%."]]),
      bullet([["Unstated details ", true], ["(distribution, clients, sequences, DQN design) had to be filled in; each choice is documented."]]),
    ],
  }],
});
Packer.toBuffer(doc).then(b => fs.writeFileSync("examiner_note_kaur2024_ton_iot.docx", b));
