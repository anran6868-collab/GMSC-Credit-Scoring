import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = process.cwd();
const publicMode = process.argv.includes("--public");
const sourcePath = path.join(projectRoot, "reports", "stage5_tables", "workbook_tables.json");
const outputPath = path.join(projectRoot, "reports", "excel", publicMode ? "stage5_model_comparison_public.xlsx" : "stage5_model_comparison.xlsx");
const previewDir = path.join(projectRoot, "reports", "stage5_workbook_previews");
const sourcePayload = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const privateSheets = new Set(["SHAP个案", "解释客户"]);
const privateColumns = new Set(["source_row_id", "row_position", "raw_feature_value", "model_input_value"]);
const payload = publicMode ? sourcePayload.filter((item) => !privateSheets.has(item.sheet)) : sourcePayload;
if (publicMode && payload.some((item) => item.columns.some((column) => privateColumns.has(column)))) {
  throw new Error("仓库版工作簿仍包含逐行客户字段，已停止生成。");
}

const workbook = Workbook.create();
const fontFamily = "Arial";
const headerFill = "#264A60";
const accentFill = "#DCE8EF";

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function safeValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

for (let sheetIndex = 0; sheetIndex < payload.length; sheetIndex += 1) {
  const item = payload[sheetIndex];
  const sheet = workbook.worksheets.add(item.sheet);
  sheet.showGridLines = false;
  sheet.tabColor = sheetIndex === 0 ? "#264A60" : (sheetIndex < 4 ? "#5E879D" : "#A8BBC5");
  const rows = item.rows.map((row) => row.map(safeValue));
  const columns = item.columns;
  const lastColumn = columnName(Math.max(columns.length - 1, 0));
  const lastRow = 5 + rows.length;

  sheet.getRange("A2").values = [[item.sheet]];
  sheet.getRange("A2").format.font = { name: fontFamily, size: 14, bold: true, color: "#1F2933" };
  sheet.getRange(`A3:${lastColumn}3`).format.borders = { bottom: { style: "thin", color: "#9FB3C1" } };
  if (sheetIndex === 0) {
    sheet.getRange("A4").values = [["验证集用于模型比较；测试集仅在模型与结论冻结后进行一次审计。"]];
    sheet.getRange("A4").format.font = { name: fontFamily, size: 10, italic: true, color: "#52606D" };
  }
  sheet.getRange(`A5:${lastColumn}5`).values = [columns];
  if (rows.length) sheet.getRange(`A6:${lastColumn}${lastRow}`).values = rows;
  const used = sheet.getRange(`A5:${lastColumn}${Math.max(lastRow, 5)}`);
  used.format.font = { name: fontFamily, size: 10, color: "#1F2933" };
  used.format.verticalAlignment = "center";
  sheet.getRange(`A5:${lastColumn}5`).format = {
    fill: headerFill,
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "medium", color: "#FFFFFF" } },
  };
  if (rows.length) {
    sheet.getRange(`A6:${lastColumn}${lastRow}`).format.borders = {
      bottom: { style: "thin", color: "#E4E7EB" },
    };
  }
  for (let col = 0; col < columns.length; col += 1) {
    const letter = columnName(col);
    const samples = [columns[col], ...rows.slice(0, 100).map((row) => String(row[col] ?? ""))];
    const longest = Math.max(...samples.map((value) => value.length));
    const name = String(columns[col]).toLowerCase();
    let width = Math.min(Math.max(longest + 2, 11), 34);
    if (name === "parameters") width = 58;
    if (name === "label_usage") width = 50;
    if (name === "best_iterations") width = 30;
    if (name === "feature" || name === "feature_cn") width = Math.min(Math.max(longest + 2, 18), 40);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = width;
    if (name === "parameters" || name === "label_usage") {
      sheet.getRange(`${letter}6:${letter}${Math.max(lastRow, 6)}`).format.wrapText = true;
    }
    if (/auc|\bks\b|pr_auc|precision|recall|\bf1\b|brier|log_loss|slope|intercept|psi|threshold|probability|pd|importance|error/.test(name)) {
      sheet.getRange(`${letter}6:${letter}${Math.max(lastRow, 6)}`).format.numberFormat = "0.000000";
    } else if (/rows|count|tree_count|nodes|leaves|feature_count|candidate_id|step/.test(name)) {
      sheet.getRange(`${letter}6:${letter}${Math.max(lastRow, 6)}`).format.numberFormat = "#,##0";
    } else if (/seconds|bytes|ms_per/.test(name)) {
      sheet.getRange(`${letter}6:${letter}${Math.max(lastRow, 6)}`).format.numberFormat = "#,##0.00";
    }
  }
  if (columns.some((name) => ["parameters", "label_usage"].includes(String(name).toLowerCase()))) {
    sheet.getRange(`A6:${lastColumn}${Math.max(lastRow, 6)}`).format.autofitRows();
  }
  if (rows.length > 20) sheet.freezePanes.freezeRows(5);
  if (columns.length > 8) sheet.freezePanes.freezeColumns(1);
  if (rows.length) {
    const table = sheet.tables.add(`A5:${lastColumn}${lastRow}`, true, `Stage5Table${sheetIndex + 1}`);
    table.style = "TableStyleMedium2";
    table.showBandedColumns = false;
  }
}

const summary = workbook.worksheets.getItem("总览");
if (payload[0]?.rows?.length) {
  const chart = summary.charts.add("bar", [
    summary.getRange("A5:A8"),
    summary.getRange("B5:B8"),
    summary.getRange("C5:C8"),
    summary.getRange("E5:E8"),
    summary.getRange("K5:K8"),
  ]);
  chart.title = "验证集三模型关键指标";
  chart.titleTextStyle.typeface = fontFamily;
  chart.legend = { position: "top", textStyle: { typeface: fontFamily } };
  chart.xAxis = { axisType: "textAxis", textStyle: { typeface: fontFamily } };
  chart.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false, textStyle: { typeface: fontFamily } };
  chart.setPosition("A11", "J29");
}

workbook.recalculate();
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const overview = await workbook.inspect({
  kind: "table",
  range: "总览!A1:N12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 14,
});
console.log(overview.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
for (const item of payload) {
  const preview = await workbook.render({ sheetName: item.sheet, autoCrop: "all", scale: 1, format: "png" });
  const safeName = item.sheet.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, previewDir, sheets: payload.length }));
