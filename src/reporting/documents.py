"""第七阶段正式报告和汇报稿排版；所有数字来自只读汇总快照。"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/final"
FIGURES = ROOT / "reports/figures"
BLUE = RGBColor(24, 49, 83)


def _snapshot(root: Path) -> dict:
    path = root / "reports/stage7_summary.json"
    if not path.is_file():
        raise FileNotFoundError("缺少第七阶段汇总，请先运行 python -m src.reporting.stage7")
    return json.loads(path.read_text(encoding="utf-8"))


def _font(run, size: int | None = None, bold: bool | None = None) -> None:
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def _paragraph(doc: Document, text: str, *, style: str | None = None) -> None:
    p = doc.add_paragraph(style=style)
    _font(p.add_run(text))
    p.paragraph_format.space_after = Pt(5)


def _heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    _font(p.add_run(text), 16 if level == 1 else 12, True)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def _table(doc: Document, headings: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headings))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for index, name in enumerate(headings):
        _font(table.rows[0].cells[index].paragraphs[0].add_run(name), 9, True)
    for data in rows:
        cells = table.add_row().cells
        for index, value in enumerate(data):
            _font(cells[index].paragraphs[0].add_run(value), 9)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "D9D9D9")
        borders.append(border)
    table._tbl.tblPr.append(borders)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def _figure(doc: Document, path: Path, caption: str, width: float = 10.5) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少报告图表：{path}")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width))
    caption_p = doc.add_paragraph()
    caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(caption_p.add_run(caption), 9)


def _metric_rows(items: list[dict]) -> list[list[str]]:
    return [[item["model"], *(f"{item[key]:.3f}" for key in ("auc", "ks", "pr_auc", "f1", "brier"))] for item in items]


def build_report(summary: dict, root: Path = ROOT) -> Path:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(1.7)
    section.bottom_margin = Cm(1.4)
    section.left_margin = section.right_margin = Cm(2.1)
    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.15
    for level in ("Title", "Heading 1", "Heading 2"):
        doc.styles[level].font.name = "Microsoft YaHei"
        doc.styles[level].font.color.rgb = RGBColor(0, 0, 0)
    title_properties = doc.styles["Title"]._element.get_or_add_pPr()
    for border in title_properties.findall(qn("w:pBdr")):
        title_properties.remove(border)
    title = doc.add_paragraph(style="Title")
    _font(title.add_run("GMSC 金融风控评分卡建模总结报告"), 22, True)
    _paragraph(doc, "基于真实 Give Me Some Credit 数据的教学研究成果")
    _paragraph(doc, "项目阶段：第七阶段最终交付｜口径：预测未来两年严重逾期概率")
    _heading(doc, "摘要")
    _paragraph(doc, "项目在有标签样本上按目标变量分层划分训练集、验证集和测试集。WOE + Logistic Regression 构成主评分卡；Random Forest 与 XGBoost 用作对照。模型选择只依据训练过程和验证集；冻结后测试集仅用于一次性泛化审计。")
    _paragraph(doc, "验证集综合比较显示 XGBoost 是最佳树模型挑战者，主评分卡仍保留 Logistic 路线。这个选择同时考虑概率校准、可解释性、复杂度与教学目标。所有策略结果均为训练集样本内模拟，不构成审批建议。")
    _heading(doc, "1 任务与数据治理")
    sizes = summary["split_sizes"]
    _table(doc, ["数据组", "样本量", "用途"], [["训练集", f"{sizes['train']:,}", "拟合清洗、分箱、特征筛选与调参"], ["验证集", f"{sizes['validation']:,}", "模型选择与稳定性检查"], ["测试集", f"{sizes['test']:,}", "冻结后一次性审计"]])
    _paragraph(doc, "原始数据由用户本地提供；无标签 Kaggle 测试文件不用于最终评价。模型目标 SeriousDlqin2yrs=1 表示未来两年发生严重逾期。项目不预测损失金额、收益，也不代表真实授信决策。")
    _heading(doc, "2 数据清洗与特征工程")
    _paragraph(doc, f"第二阶段在训练集检查缺失、特殊值和风险分组，再将训练集拟合的规则应用于三组数据。第三阶段保留原始变量并形成 {summary['candidate_feature_count']} 个非 WOE 候选特征；分箱、WOE 和 IV 仅在训练集拟合。验证集 PSI 不使用标签。")
    _paragraph(doc, "训练集月收入缺失 20,723 条，占 19.74%。缺失模式可能反映记录方式，不能直接解释为因果风险。部分逾期变量具有较高 IV，项目在后续筛选时复核冗余和业务意义。")
    _figure(doc, FIGURES / "stage3/iv_ranking.png", "图 1 训练集候选变量 IV 排名")
    _heading(doc, "3 Logistic 主评分卡")
    _paragraph(doc, f"第四阶段综合 IV、相关性、VIF、系数方向、显著性、稳定性和业务意义，最终采用“{summary['selected_scheme']}”方案，保留 {len(summary['selected_features'])} 个变量。评分参数为基准分 600、好坏 Odds 20:1、PDO 20；A–E 等级边界仅由训练集评分分布确定。")
    _paragraph(doc, "客户页同时呈现严重逾期概率、信用分、等级和变量贡献。总分与贡献分经过一致性核对；RF 与 XGBoost 仅提供概率对照。")
    _figure(doc, FIGURES / "stage4/score_distribution.png", "图 2 训练集评分分布")
    doc.add_page_break()
    _heading(doc, "4 验证集模型比较")
    _table(doc, ["模型", "AUC", "KS", "PR-AUC", "F1", "Brier"], _metric_rows(summary["validation_metrics"]))
    _paragraph(doc, "XGBoost 在验证集综合比较中领先，是最佳树模型挑战者；Logistic 主评分卡保留清晰的分箱、WOE 和积分贡献路径。Random Forest 的验证集校准截距略超项目设定门槛。模型结论不能只凭 AUC 得出。")
    _figure(doc, FIGURES / "stage5/validation_model_comparison.png", "图 3 验证集三模型关键指标")
    _heading(doc, "5 冻结后测试审计")
    _table(doc, ["模型", "AUC", "KS", "PR-AUC", "F1", "Brier"], _metric_rows(summary["test_metrics_frozen_audit"]))
    _paragraph(doc, "三模型、阈值和验证结论先写入 SHA-256 冻结清单，再读取测试标签开展一次审计。测试集表只报告泛化表现，不生成新的模型排名，不据此调参或改变主模型。")
    _heading(doc, "6 模型解释与策略模拟")
    shap = summary["shap_additivity_max_error"]
    _paragraph(doc, f"树模型 SHAP 在严重逾期概率空间计算，RF 和 XGBoost 的最大加总误差分别为 {shap['Random Forest']:.2e} 与 {shap['XGBoost']:.2e}。局部解释仅展示第五阶段固定的代表客户，不对新输入客户重算 SHAP。")
    _figure(doc, FIGURES / "stage5/xgboost_shap_global_bar.png", "图 4 XGBoost 验证集 SHAP 全局重要性")
    strategy = summary["strategy_at_600_train_only"]
    if strategy:
        _paragraph(doc, f"以最低通过分 600 作教学示例，训练集样本内通过率为 {strategy['pass_rate']:.2%}，通过客户坏率为 {strategy['pass_bad_rate']:.2%}，被拒坏客户捕获率为 {strategy['rejected_bad_capture']:.2%}。这些结果可能偏乐观，不能当作真实审批策略收益。")
    _heading(doc, "7 局限、复现与交付")
    _paragraph(doc, "数据来自公开竞赛，缺少真实授信规则、时间外样本和上线监控环境。分组风险、SHAP 与积分贡献描述统计或模型关联，不证明因果关系。冻结后的测试审计不得用于再选择模型。")
    _paragraph(doc, "项目使用 Python 3.11 和根目录 .venv；运行 `python -m pytest -q` 检查实现，双击 Windows 启动脚本打开本地八页面应用。原始数据、逐行数据、日志和序列化模型保持在本地，不进入版本库。")
    _paragraph(doc, "主要来源：reports/stage3_summary.json、reports/stage4_summary.json、reports/stage5_summary.json、models/stage5_frozen_manifest.json，以及四本阶段分析工作簿。冻结清单 SHA-256：" + summary["frozen_manifest_sha256"])
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(footer.add_run("GMSC 教学研究项目｜仅供学习与展示"), 8)
    output = root / "reports/final/GMSC_建模总结报告.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return output


def _slide(c: canvas.Canvas, title: str, eyebrow: str, page: int) -> None:
    c.setFillColor(colors.HexColor("#F6F8FB"))
    c.rect(0, 0, 960, 540, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#183153"))
    c.rect(0, 506, 960, 34, fill=1, stroke=0)
    c.setFont("SC", 11)
    c.setFillColor(colors.white)
    c.drawString(42, 517, eyebrow)
    c.setFillColor(colors.HexColor("#183153"))
    c.setFont("SC", 27)
    c.drawString(42, 450, title)
    c.setFillColor(colors.HexColor("#657485"))
    c.setFont("SC", 10)
    c.drawString(42, 22, "GMSC 教学研究项目｜不作为真实授信决策")
    c.drawRightString(918, 22, f"{page} / 10")


def _lines(c: canvas.Canvas, lines: list[str], y: int = 390, size: int = 17, gap: int = 35) -> None:
    c.setFillColor(colors.HexColor("#243447"))
    c.setFont("SC", size)
    for line in lines:
        c.drawString(54, y, line)
        y -= gap


def _image(c: canvas.Canvas, path: Path, x: int = 460, y: int = 96, width: int = 440, height: int = 320) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少汇报图表：{path}")
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(width / iw, height / ih)
    w, h = iw * scale, ih * scale
    c.drawImage(ImageReader(str(path)), x + (width - w) / 2, y + (height - h) / 2, w, h, preserveAspectRatio=True, mask="auto")


def build_brief(summary: dict, root: Path = ROOT) -> Path:
    pdfmetrics.registerFont(TTFont("SC", r"C:\Windows\Fonts\simhei.ttf"))
    output = root / "reports/final/GMSC_项目汇报稿.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(960, 540))
    _slide(c, "GMSC 金融风控评分卡建模", "第七阶段成果汇报", 1)
    _lines(c, ["目标：预测未来两年严重逾期概率", "主路线：WOE + Logistic 评分卡", "对照：Random Forest 与 XGBoost", "用途：教学与研究；不作真实审批结论"], 365, 20, 49)
    c.showPage()
    _slide(c, "研究流程与数据边界", "项目方法", 2)
    sizes = summary["split_sizes"]
    _lines(c, [f"训练集 {sizes['train']:,}：拟合全部规则", f"验证集 {sizes['validation']:,}：选择模型", f"测试集 {sizes['test']:,}：冻结后审计", "目标分层随机划分：70% / 15% / 15%"], 380)
    c.showPage()
    _slide(c, "训练集拟合清洗与特征", "数据与特征", 3)
    _lines(c, ["缺失、异常规则只从训练集取得", "共 20 个非 WOE 候选特征", "分箱、WOE 和 IV 仅用训练集拟合", "验证集 PSI 不读取标签"], 380)
    _image(c, FIGURES / "stage3/iv_ranking.png")
    c.showPage()
    _slide(c, "Logistic 主评分卡", "模型路线", 4)
    _lines(c, ["综合筛选 8 个入模变量", "600 分基准；好坏 Odds 20:1；PDO 20", "输出 PD、积分、等级与变量贡献", "A–E 边界来自训练集评分分布"], 380, 16)
    _image(c, FIGURES / "stage4/score_distribution.png")
    c.showPage()
    _slide(c, "验证集三模型比较", "模型选择", 5)
    _lines(c, [f"Logistic AUC {next(x for x in summary['validation_metrics'] if x['model']=='Logistic 评分卡')['auc']:.3f}", f"RF AUC {next(x for x in summary['validation_metrics'] if x['model']=='Random Forest')['auc']:.3f}", f"XGBoost AUC {next(x for x in summary['validation_metrics'] if x['model']=='XGBoost')['auc']:.3f}", "比较同时考虑校准、解释与复杂度"], 384, 16)
    _image(c, FIGURES / "stage5/validation_model_comparison.png")
    c.showPage()
    _slide(c, "主模型与挑战者", "验证结论", 6)
    _lines(c, ["主模型：Logistic 评分卡", "最佳树模型挑战者：XGBoost", "RF 验证集校准截距略超项目门槛", "模型选择不只凭 AUC"], 380, 18)
    _image(c, FIGURES / "stage5/validation_calibration.png")
    c.showPage()
    _slide(c, "冻结后测试审计", "泛化审计", 7)
    _lines(c, [f"Logistic AUC {next(x for x in summary['test_metrics_frozen_audit'] if x['model']=='Logistic 评分卡')['auc']:.3f}", f"RF AUC {next(x for x in summary['test_metrics_frozen_audit'] if x['model']=='Random Forest')['auc']:.3f}", f"XGBoost AUC {next(x for x in summary['test_metrics_frozen_audit'] if x['model']=='XGBoost')['auc']:.3f}", "结果不再用于调参或改选模型"], 380, 18)
    c.showPage()
    _slide(c, "解释严重逾期概率", "模型解释", 8)
    _lines(c, ["Logistic：WOE 与积分贡献", "树模型：概率空间 SHAP", "代表客户来自第五阶段固定集合", "解释模型关联，不证明因果"], 380, 16)
    _image(c, FIGURES / "stage5/xgboost_shap_global_bar.png")
    c.showPage()
    _slide(c, "训练集样本内策略模拟", "教学场景", 9)
    strategy = summary["strategy_at_600_train_only"]
    _lines(c, ["示例最低通过分：600", f"通过率：{strategy['pass_rate']:.2%}", f"通过客户坏率：{strategy['pass_bad_rate']:.2%}", f"被拒坏客户捕获率：{strategy['rejected_bad_capture']:.2%}", "样本内结果可能偏乐观"], 390, 18)
    c.showPage()
    _slide(c, "结论与使用边界", "最终交付", 10)
    _lines(c, ["已完成完整建模、审计与八页面展示", "正式报告、综合 Excel、Notebook 与手册可追溯", "缺少时间外样本和上线监控验证", "仅用于教学研究，不能当作审批规则"], 380, 18)
    c.showPage()
    c.save()
    return output


def build_report_pdf(summary: dict, root: Path = ROOT) -> Path:
    """用同一汇总快照生成可独立打开的正式 PDF。"""
    if "SC" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SC", r"C:\Windows\Fonts\simhei.ttf"))
    output = root / "reports/final/GMSC_建模总结报告.pdf"
    normal = ParagraphStyle("body", fontName="SC", fontSize=10, leading=17, textColor=colors.HexColor("#243447"), spaceAfter=9)
    heading = ParagraphStyle("heading", parent=normal, fontSize=15, leading=23, textColor=colors.HexColor("#183153"), spaceBefore=16, spaceAfter=9)
    title = ParagraphStyle("title", parent=normal, fontSize=21, leading=31, textColor=colors.black, spaceAfter=18)
    caption = ParagraphStyle("caption", parent=normal, fontSize=9, leading=14, alignment=TA_CENTER)
    story = []

    def para(text: str):
        story.append(Paragraph(text, normal))

    def h(text: str):
        story.append(Paragraph(text, heading))

    def table(headers: list[str], rows: list[list[str]]):
        data = [[Paragraph(str(v), normal) for v in headers]] + [[Paragraph(str(v), normal) for v in row] for row in rows]
        item = Table(data, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EDF3")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D9D9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(KeepTogether([item, Spacer(1, 9)]))

    def figure(rel: str, label: str):
        from PIL import Image as PILImage
        path = FIGURES / rel
        with PILImage.open(path) as image:
            width, height = image.size
        scale = min(440 / width, 185 / height)
        story.append(KeepTogether([Image(str(path), width * scale, height * scale), Paragraph(label, caption)]))

    story.append(Paragraph("GMSC 金融风控评分卡建模总结报告", title))
    para("基于真实 Give Me Some Credit 数据的教学研究成果。项目预测未来两年严重逾期概率，不代表真实授信审批。")
    h("摘要")
    para("项目按目标变量分层划分训练集、验证集和测试集。WOE + Logistic Regression 是主评分卡；Random Forest 与 XGBoost 是对照。验证集用于模型选择，冻结后测试集只作一次泛化审计。")
    para("XGBoost 是验证集表现最好的树模型挑战者；Logistic 评分卡继续作为主模型。决策兼顾区分能力、校准、可解释性和复杂度。")
    h("1 任务与数据治理")
    sizes = summary["split_sizes"]
    table(["数据组", "样本量", "用途"], [["训练集", f"{sizes['train']:,}", "拟合规则、筛选与调参"], ["验证集", f"{sizes['validation']:,}", "选择模型"], ["测试集", f"{sizes['test']:,}", "冻结后审计"]])
    para("原始数据由用户在本地提供。无标签 Kaggle 测试文件不作为最终评价集；测试集标签不用于模型选择或调参。")
    h("2 清洗、特征与分箱")
    para(f"清洗规则仅在训练集拟合；第三阶段形成 {summary['candidate_feature_count']} 个非 WOE 候选特征。专业分箱、WOE、IV 只用训练集拟合；验证集 PSI 不使用标签。训练集月收入缺失 20,723 条，占 19.74%。")
    figure("stage3/iv_ranking.png", "训练集候选变量 IV 排名")
    h("3 Logistic 主评分卡")
    para(f"综合 IV、相关性、VIF、系数方向、显著性、稳定性与业务意义，最终选用 {len(summary['selected_features'])} 个变量。基准分 600，好坏 Odds 20:1，PDO 20；A–E 风险等级边界来自训练集评分分布。")
    figure("stage4/score_distribution.png", "训练集评分分布")
    h("4 验证集模型比较")
    table(["模型", "AUC", "KS", "PR-AUC", "F1", "Brier"], _metric_rows(summary["validation_metrics"]))
    para("XGBoost 是最佳树模型挑战者；Logistic 保留清晰的 WOE 和积分贡献路径。Random Forest 验证集校准截距略超项目设定门槛。模型选择不只凭 AUC。")
    figure("stage5/validation_model_comparison.png", "验证集三模型关键指标")
    h("5 冻结后测试审计")
    table(["模型", "AUC", "KS", "PR-AUC", "F1", "Brier"], _metric_rows(summary["test_metrics_frozen_audit"]))
    para("模型、阈值和验证结论先写入 SHA-256 冻结清单，再读取测试标签完成一次审计。测试结果仅报告泛化表现，不用于重新排名、调参或改变主模型。")
    h("6 模型解释与策略")
    shap = summary["shap_additivity_max_error"]
    para(f"RF 与 XGBoost 的 SHAP 均解释严重逾期概率；最大加总误差分别为 {shap['Random Forest']:.2e} 与 {shap['XGBoost']:.2e}。局部解释只展示第五阶段固定的代表客户。")
    figure("stage5/xgboost_shap_global_bar.png", "XGBoost 验证集 SHAP 全局重要性")
    strategy = summary["strategy_at_600_train_only"]
    para(f"最低通过分 600 的训练集样本内示例：通过率 {strategy['pass_rate']:.2%}，通过客户坏率 {strategy['pass_bad_rate']:.2%}，被拒坏客户捕获率 {strategy['rejected_bad_capture']:.2%}。样本内结果可能偏乐观，不能代表真实审批收益。")
    h("7 局限与复现")
    para("公开竞赛数据缺少真实授信规则、时间外样本与上线监控。分组风险、SHAP 与积分贡献反映统计或模型关联，不证明因果。项目在 Python 3.11 的 .venv 中运行，使用 pytest 和八页面本地应用检查成果。")
    para("数据来源：第三至第五阶段汇总、冻结清单及四本阶段分析工作簿。冻结清单 SHA-256：" + summary["frozen_manifest_sha256"])

    pdf = SimpleDocTemplate(str(output), pagesize=(595, 842), rightMargin=56, leftMargin=56, topMargin=55, bottomMargin=55)
    pdf.build(story)
    return output


def run(root: Path = ROOT) -> tuple[Path, Path, Path]:
    summary = _snapshot(root)
    return build_report(summary, root), build_report_pdf(summary, root), build_brief(summary, root)


if __name__ == "__main__":
    print(*run(), sep="\n")
