"""Render the reviewed Markdown layout with 12 body pages and flowing appendices.

This document utility requires reportlab and pypdf. It does not run experiments.
Tables wrap in their cells; long URLs are linked by descriptive labels.
"""
import argparse
from html import escape
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepInFrame, Flowable,
)
from pypdf import PdfReader


class BodyPage(Flowable):
    """Record the scale used to fit a deliberate body page; reject illegibility."""
    def __init__(self, content, width, height, page_number, scales):
        super().__init__()
        self.box = KeepInFrame(width, height, content, mode="shrink", hAlign="LEFT", vAlign="TOP")
        self.width, self.height = width, height
        self.number, self.scales = page_number, scales

    def wrap(self, width, height):
        self.box.canv = self.canv
        self.box.wrap(width, height)
        scale = getattr(self.box, "_scale", 1)
        self.scales[str(self.number)] = scale
        if scale > 1.15:
            raise ValueError(f"Body page {self.number} needs excessive shrink {scale:.3f}; shorten the layout")
        return self.width, self.height

    def draw(self):
        self.box.drawOn(self.canv, 0, self.height - self.box.height)


def inline(text):
    text = escape(text)
    text = re.sub(r'!\[([^]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[([^]]+)\]\(([^)]+)\)',
                  lambda m: f'<link href="{m[2]}" color="#245A83">{m[1]}</link>'
                  if m[2].startswith('https://') else m[1], text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = text.replace('`', '').replace('$', '')
    text = text.replace('\\mathcal A', 'A').replace('\\in', '∈').replace('\\{','{').replace('\\}', '}')
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = text.replace('\\bar m', 'm̄').replace('\\Delta W', 'ΔW').replace('\\theta', 'θ')
    text = text.replace('\\mathcal L', 'L')
    return text


def formula(block):
    """Equivalent, readable formulas; definitions remain in the adjacent prose."""
    if 'w_{t,d}' in block:
        return 'w(t,d) = (1 + ln f(t,d)) × [ln ((1 + N训练) / (1 + df(t))) + 1]，f(t,d) &gt; 0'
    if 'softmax' in block:
        return 'p(x) = softmax(Wc · hpool(x) + bc)，预测标签 = argmax p(c | x)'
    if 'W_0' in block:
        return 'W = W<sub>0</sub> + ΔW = W<sub>0</sub> + (α/r) BA'
    if 'answer' in block:
        return 'L答案 = - (1 / |A|) ∑[t∈A] ln p<sub>θ</sub>(y<sub>t</sub> | x, y<sub>&lt;t</sub>)'
    if '有效' in block:
        return 'B有效 = B微批量 × K累积 × G设备数 = 1 × 16 × 1 = 16'
    if 'Precision}_c' in block:
        return '精确率 Pc = TPc / (TPc + FPc)<br/>召回率 Rc = TPc / (TPc + FNc)<br/>类别 F1c = 2PcRc / (Pc + Rc) = 2TPc / (2TPc + FPc + FNc)'
    if 'Weighted' in block:
        return 'Macro-F1 = (F1负面 + F1正面) / 2<br/>Weighted-F1 = ∑[c=0,1] (nc/N) F1c'
    if 'InvalidRate' in block:
        return 'Accuracy = 预测正确的评论条数 / N<br/>非法输出率 = 未严格输出0或1的评论条数 / N'
    if 'mathrm{SD}' in block:
        return '均值 m_mean = (1/S) ∑ m<sub>s</sub><br/>样本 SD = √[∑(m<sub>s</sub> - m_mean)² / (S - 1)]，S = 3'
    if '6991' in block:
        return 'P(三种子平均F1：MacBERT &gt; LoRA) ≈ 6991/10000 = 0.6991 ≈ 70%'
    if '\\bar t' in block:
        return '平均延迟 = ∑t<sub>i</sub> / n；串行吞吐 = n / (∑t<sub>i</sub> / 1000) 条/秒，n = 200'
    raise ValueError('Unrecognized formula: '+block)


def render(source, target, font_path):
    pdfmetrics.registerFont(TTFont('CN', font_path))
    pdfmetrics.registerFontFamily('CN', normal='CN', bold='CN', italic='CN', boldItalic='CN')
    pagew,pageh=A4
    width=pagew-76
    styles={
        'p':ParagraphStyle('p',fontName='CN',fontSize=10.2,leading=15.1,spaceAfter=5,wordWrap='CJK'),
        'h1':ParagraphStyle('h1',fontName='CN',fontSize=17,leading=22,spaceAfter=10,textColor=colors.HexColor('#173B58')),
        'h2':ParagraphStyle('h2',fontName='CN',fontSize=14,leading=19,spaceBefore=5,spaceAfter=8,textColor=colors.HexColor('#173B58')),
        'h3':ParagraphStyle('h3',fontName='CN',fontSize=11.2,leading=16,spaceBefore=7,spaceAfter=5,textColor=colors.HexColor('#245A83')),
        'table':ParagraphStyle('table',fontName='CN',fontSize=9.4,leading=12.9,wordWrap='CJK'),
        'code':ParagraphStyle('code',fontName='CN',fontSize=8.7,leading=12,wordWrap='CJK',backColor=colors.HexColor('#F1F5F8'),spaceAfter=4),
        'formula':ParagraphStyle('formula',fontName='CN',fontSize=10.5,leading=15.6,alignment=TA_CENTER,spaceBefore=4,spaceAfter=7),
    }
    for name in ('h1','h2','h3'):
        styles[name].keepWithNext=True
    def parse(text):
        lines=text.splitlines(); out=[]; i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line or line=='---': i+=1;continue
            if line.startswith('$$'):
                i+=1; block=[]
                while i<len(lines) and not lines[i].strip().startswith('$$'):
                    block.append(lines[i]); i+=1
                out.append(Paragraph(formula('\n'.join(block)),styles['formula']));i+=1;continue
            if line.startswith('```'):
                i+=1
                while i<len(lines) and not lines[i].strip().startswith('```'):
                    if lines[i].strip(): out.append(Paragraph(escape(lines[i]),styles['code']))
                    i+=1
                i+=1;continue
            if line.startswith('|'):
                data=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    cells=lines[i].strip().strip('|').split('|')
                    if not all(re.fullmatch(r'\s*:?-+:?\s*',c) for c in cells):
                        data.append([c.strip() for c in cells])
                    i+=1
                n=max(map(len,data)); data=[r+['']*(n-len(r)) for r in data]
                sizes=[]
                for j in range(n):
                    lengths=[len(re.sub(r'`|\*','',row[j])) for row in data]
                    sizes.append(min(40,max(9,max(lengths)**0.7*2)))
                if n==4 and any(word in data[0][-1] for word in ('工作量','工时')): sizes=[23,42,38,11]
                if n==4 and '归因' in ''.join(data[0]): sizes=[20,32,32,30]
                table=Table([[Paragraph(inline(c),styles['table']) for c in r] for r in data],
                            colWidths=[width*x/sum(sizes) for x in sizes],repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E5EFF6')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFC')]),
                    ('LINEBELOW',(0,0),(-1,0),0.7,colors.HexColor('#7094AE')),
                    ('LINEBELOW',(0,1),(-1,-1),0.25,colors.HexColor('#D8E3EB')),
                    ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                    ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
                ]));out.extend([table,Spacer(1,6)]);continue
            if line.startswith('!['):
                match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
                path=(source.parent/match[2]).resolve()
                if path.suffix=='.svg':path=path.with_suffix('.png')
                img=Image(str(path)); ratio=img.imageHeight/img.imageWidth
                maxheight=190 if '架构' in path.name else 172
                iw=min(width,maxheight/ratio);out.extend([Image(str(path),width=iw,height=iw*ratio),Spacer(1,5)])
                i+=1;continue
            heading=re.match(r'^(#{1,3})\s+(.+)',line)
            if heading:
                out.append(Paragraph(inline(heading[2]),styles['h'+str(len(heading[1]))]));i+=1;continue
            if line.startswith('>'):line=line.lstrip('> ').strip()
            if line.startswith('- '):line='• '+line[2:]
            out.append(Paragraph(inline(line),styles['p']));i+=1
        return out
    text=source.read_text(encoding='utf-8')
    body,appendix=text.split('<!-- APPENDIX -->')
    pages=body.split('<!-- PAGEBREAK -->')
    assert len(pages)==12
    scales={};story=[]
    for n,page in enumerate(pages,1):
        story.append(BodyPage(parse(page),width,pageh-87,n,scales));story.append(PageBreak())
    for n,section in enumerate(appendix.split('<!-- PAGEBREAK -->')):
        if n:
            story.append(PageBreak() if section.lstrip().startswith('# 附录G') else Spacer(1,16))
        story.extend(parse(section))
    def decoration(canvas,doc):
        canvas.saveState();canvas.setFont('CN',8)
        canvas.setFillColor(colors.HexColor('#537085'))
        canvas.drawString(38,pageh-24,'自然语言处理课程项目 | 中文酒店评论情感分析 | 2026-10-30')
        canvas.drawRightString(pagew-38,23,f'{"正文" if doc.page<=12 else "附录"} · {doc.page if doc.page<=12 else doc.page-12}')
        canvas.restoreState()
    target.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=38,rightMargin=38,
                          topMargin=39,bottomMargin=36,title='中文酒店评论情感分析 - 课程项目报告',author='刘广徽、杨函齐、李嘉谊林')
    doc.build(story,onFirstPage=decoration,onLaterPages=decoration)
    pdf=PdfReader(target)
    status={'body_pages':12,'total_pages':len(pdf.pages),'body_page_shrink_factors':scales,
            'minimum_effective_body_font_pt':10.2/max(scales.values()),'visual_review':'pending'}
    target.with_suffix('.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/报告初稿/20261029-v6/PDF排版稿.md'))
    parser.add_argument('--output',type=Path,default=Path('output/pdf/课程项目报告-v6.pdf'))
    parser.add_argument('--font',default='C:/Windows/Fonts/simhei.ttf')
    args=parser.parse_args()
    render(args.source.resolve(),args.output.resolve(),args.font)
