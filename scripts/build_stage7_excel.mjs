import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const payload = JSON.parse(await fs.readFile(path.join(root, "reports/stage7_workbook_tables.json"), "utf8"));
const workbook = Workbook.create();

function columnName(index) {
  let n = index + 1;
  let result = "";
  while (n > 0) {
    result = String.fromCharCode(65 + ((n - 1) % 26)) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

for (const [index, item] of payload.entries()) {
  const sheet = workbook.worksheets.add(item.sheet);
  sheet.showGridLines = false;
  sheet.tabColor = index === 0 ? "#183153" : index === payload.length - 1 ? "#6A8296" : "#9BB4C9";
  const columns = item.columns;
  const rows = item.rows.map(row => row.map(value => value === null || value === undefined ? null : value));
  const lastColumn = columnName(columns.length - 1);
  const lastRow = 5 + rows.length;
  sheet.getRange("A2").values = [[item.sheet]];
  sheet.getRange("A2").format.font = { name: "Microsoft YaHei", size: 15, bold: true, color: "#183153" };
  sheet.getRange("A4").values = [[item.source ?? "验证集用于模型比较；测试集仅用于冻结后审计。"]];
  sheet.getRange("A4").format.font = { name: "Microsoft YaHei", size: 10, color: "#52606D" };
  sheet.getRange(`A5:${lastColumn}5`).values = [columns];
  if (rows.length) sheet.getRange(`A6:${lastColumn}${lastRow}`).values = rows;
  const used = sheet.getRange(`A5:${lastColumn}${Math.max(lastRow, 5)}`);
  used.format.font = { name: "Microsoft YaHei", size: 10, color: "#1F2933" };
  used.format.verticalAlignment = "center";
  sheet.getRange(`A5:${lastColumn}5`).format = {
    fill: "#183153",
    font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  for (let col = 0; col < columns.length; col += 1) {
    const letter = columnName(col);
    const sample = [columns[col], ...rows.slice(0, 100).map(row => String(row[col] ?? ""))];
    const width = Math.min(Math.max(...sample.map(value => value.length)) + 2, 42);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = Math.max(width, 13);
    const name = String(columns[col]).toLowerCase();
    if (/auc|ks|iv|psi|f1|brier|log_loss|woe|rate|pd|importance|threshold|share/.test(name)) {
      sheet.getRange(`${letter}6:${letter}${Math.max(lastRow, 6)}`).format.numberFormat = "0.000000";
    }
  }
  if (rows.length > 20) sheet.freezePanes.freezeRows(5);
  if (columns.length > 8) sheet.freezePanes.freezeColumns(1);
  if (rows.length) {
    const table = sheet.tables.add(`A5:${lastColumn}${lastRow}`, true, `Stage7Table${index + 1}`);
    table.style = "TableStyleMedium2";
    table.showBandedColumns = false;
  }
}

workbook.recalculate();
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final error scan",
});
console.log(errors.ndjson);
const output = path.join(root, "reports/excel/stage7_comprehensive_report.xlsx");
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(output);
console.log(JSON.stringify({ output, sheets: payload.length }));
