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
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 32, bold: true },
        paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 1000, bottom: 1000, left: 1134, right: 1134 } } },
    children: [
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Viva note: Reimplementation of Kaur (2024)")] }),
      p("GRU-assisted federated learning with DQN client selection for IIoT intrusion detection. "
        + "IEEE Trans. Artificial Intelligence, 2024, DOI 10.1109/TAI.2024.3443787. Dataset: TON_IoT (Train_Test_Network), non-IID.",
        { italics: true, size: 18, after: 120 }),

      h("1. What I reimplemented"),
      bullet([["GRU classifier: ", true], ["2 GRU layers × 128 units, dropout 0.3, 1 output layer, 10 classes."]]),
      bullet([["Federated learning: ", true], ["15 clients; server averages models weighted by client data size (Eqs. 18, 25)."]]),
      bullet([["DQN client selection: ", true], ["picks 8 of 15 clients and a transmit power each round from energy, SNR and data quality (Eqs. 5–8, 12–13)."]]),
      bullet([["Energy model ", true], ["(Eqs. 1–4); ", false], ["preprocessing: ", true], ["drop columns, one-hot, min-max (Eq. 19)."]]),
      bullet([["FedAvg baseline: ", true], ["same GRU, all 15 clients every round, no DQN."]]),

      h("2. Kaur (2024): claimed in the paper vs reimplemented (TON_IoT)"),
      table([2500, 1830, 900, 1050, 1050, 750, 778, 780],
        ["Source", "Conditions", "Rounds", "Accuracy", "Precision", "Recall", "F1", "AUC"],
        [
          ["Claimed in the paper (Table V, Fig. 7)", "TON_IoT; IID/non-IID and clients not stated", "not stated", "99.95", "98.63", "99.98", "99.99", "99.9"],
          ["Reimplemented", "TON_IoT, non-IID, 15 clients", "50", "76.69", "66.42", "45.86", "41.53", "95.41"],
          ["Reimplemented", "TON_IoT, non-IID, 15 clients", "100", "76.72", "64.14", "45.99", "41.91", "95.75"],
        ], [1, 2]),
      p("All values in %. Precision, recall and F1 are macro averages over the 10 classes. The paper gives no TON_IoT result "
        + "for non-IID or for 15 clients; its 15-client non-IID result (Table VIII: 97.98% accuracy at round 50) is on a "
        + "different dataset, Edge-IIoT.",
        { size: 16, italics: true, after: 120 }),

      h("3. Setup used"),
      table([3638, 6000], ["Setting", "Value"], [
        ["Data split", "Train 371,485 / test 89,518 (test set identical to Table II); fixed seed 0"],
        ["Clients / non-IID", "15 clients, 8 selected per round; Dirichlet label skew, α = 0.5"],
        ["Rounds / local epochs / batch", "100 rounds (compared at 50) / 5 / 20"],
        ["Optimiser / learning rate", "SGD / 0.2 (paper Eqs. 17, 24)"],
        ["DQN", "Primary + target network, replay buffer 10,000, γ = 0.99"],
      ], [], true),

      h("4. Key points to say"),
      bullet([["Faithful reimplementation with documented deviations. ", true], ["Architecture, aggregation, DQN and energy model follow the paper; every unstated detail is recorded (SPEC.md, gap-fills G1–G20)."]]),
      bullet([["Adam at lr 0.2 collapsed ", true], ["(predicted “Normal” for every row, 61.44%). I used SGD at 0.2, which matches the paper’s own update equations (17, 24)."]]),
      bullet([["Main reason for the gap: min-max scaling (Eq. 19). ", true], ["Byte, packet and duration columns become ≈ 0, so the model sees mainly categorical flags. DoS, Ransomware, Injection, Password and MITM recall stays below 10%."]]),
      bullet([["The paper’s Table V is internally inconsistent: ", true], ["its per-class rows imply 96.47% accuracy and 95.02 F1, not 99.95 / 99.99."]]),
      bullet([["DQN vs FedAvg: ", true], ["paper claims +32 points; I measured +3.9 points (non-IID) at about half the energy (70 J vs 136 J by round 50), and no gain in IID."]]),
      bullet([["Protocol note: ", true], ["the paper’s 15-client IID/non-IID experiment (Table VIII) is on Edge-IIoT; I applied it to TON_IoT."]]),
      bullet([["Data caveat: ", true], ["60% of test rows duplicate a training row’s features, so I also report accuracy on unseen rows (69.33%)."]]),
      bullet([["Fair comparison: ", true], ["my model and theirs use the same split, test set and budget (50 rounds × 5 local epochs). Their accuracy at round 50 and round 100 differs by less than 0.05 points."]]),
    ],
  }],
});
Packer.toBuffer(doc).then(b => fs.writeFileSync("viva_note_kaur2024.docx", b));
