# -*- coding: utf-8 -*-
"""
Anesthesia Residency Feedback Utility Evaluation System
RSA Diagnostic Baseline v3.0 with Batch Testing and ML Prediction
Full English Interface for International Deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ========================= Configuration =========================
SAVE_DIR = r"C:\Users\31217\Desktop\构建\效用"   # Adjust to your path
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== RSA Diagnostic Baseline v3.0 (Extended) ====================
class RSADiagnosticBaseline:
    def __init__(self):
        self._init_entity_dict()
        self._init_specificity_patterns_v3()
        self._init_actionability_patterns_v3()
        self.core_keywords = ["麻醉", "教学", "培训", "手术", "值班", "临床", "医师", "医生", "学员", "住院医师"]

    def _init_entity_dict(self):
        self.relevance_entities = {
            "临床操作": [
                "气管插管", "硬膜外穿刺", "腰麻", "蛛网膜下腔阻滞", "全麻诱导",
                "全麻维持", "麻醉复苏", "拔管", "中心静脉穿刺", "动脉穿刺",
                "椎管内麻醉", "臂丛阻滞", "颈丛阻滞", "坐骨神经阻滞", "股神经阻滞",
                "喉罩置入", "纤支镜引导插管", "双腔气管插管", "支气管封堵器",
                "快速顺序诱导", "清醒插管", "困难气道处理", "环甲膜穿刺",
                "术后镇痛", "病人自控镇痛", "神经阻滞镇痛", "硬膜外镇痛",
                "胸科麻醉", "心脏麻醉", "心血管麻醉", "神经外科麻醉",
                "骨科麻醉", "产科麻醉", "儿科麻醉", "老年麻醉",
                "日间手术麻醉", "急诊麻醉", "创伤麻醉", "移植麻醉",
                "肝胆外科麻醉", "泌尿外科麻醉", "普外科麻醉",
                "胸科", "麻醉"
            ],
            "药物使用": [
                "丙泊酚", "依托咪酯", "咪达唑仑", "右美托咪定", "氯胺酮",
                "芬太尼", "瑞芬太尼", "舒芬太尼", "阿芬太尼", "吗啡",
                "罗库溴铵", "顺苯磺阿曲库铵", "维库溴铵", "琥珀胆碱",
                "七氟烷", "地氟烷", "异氟烷", "氧化亚氮", "恩氟烷",
                "利多卡因", "罗哌卡因", "布比卡因", "丁卡因", "普鲁卡因",
                "阿托品", "新斯的明", "氟马西尼", "纳洛酮", "舒更葡糖钠",
                "局麻药", "肌松药", "镇静药", "镇痛药", "血管活性药"
            ],
            "监测指标": [
                "血压", "收缩压", "舒张压", "平均动脉压", "心率",
                "血氧饱和度", "脉搏氧饱和度", "呼气末二氧化碳", "PaCO2",
                "麻醉深度", "BIS", "脑电双频指数", "肌松监测", "TOF",
                "中心静脉压", "肺动脉楔压", "心输出量", "每搏量变异度",
                "动脉血气", "电解质", "血糖", "血红蛋白", "红细胞压积",
                "尿量", "出血量", "尿比重", "体温", "鼻咽温度", "直肠温度",
                "呼末CO2", "SpO2", "BP", "HR", "CVP",
                "血流动力学"
            ],
            "麻醉设备": [
                "麻醉机", "呼吸机", "监护仪", "输液泵", "注射泵",
                "喉镜", "视频喉镜", "可视喉镜", "光棒", "喉罩",
                "气管导管", "双腔管", "支气管封堵器", "吸痰管", "口咽通气道",
                "鼻咽通气道", "面罩", "回路", "钠石灰", "挥发罐",
                "穿刺针", "导管", "延长管", "三通", "过滤器"
            ],
            "并发症/不良事件": [
                "术中知晓", "术后恶心呕吐", "苏醒延迟", "呼吸抑制",
                "低血压", "高血压", "心动过缓", "心动过速", "心律失常",
                "支气管痉挛", "喉痉挛", "误吸", "反流", "局麻药中毒",
                "全脊麻", "硬膜穿破后头痛", "神经损伤", "气胸", "血肿",
                "低氧血症", "高碳酸血症", "酸中毒", "电解质紊乱"
            ],
            "非技术技能": [
                "沟通", "交流", "团队合作", "团队配合", "领导力",
                "决策", "临床决策", "情境意识", "态势感知", "压力管理",
                "时间管理", "优先级排序", "交接班", "术前访视", "术后随访",
                "与患者沟通", "与外科医生沟通", "与护士沟通", "知情同意",
                "独立值班", "值班能力", "应急处理", "突发事件",
                "值班"
            ],
            "教学评价": [
                "技能", "操作能力", "临床能力", "技术水平", "专业能力",
                "学习态度", "积极性", "主动性", "责任心", "出勤", "考勤",
                "理论知识", "实践技能", "动手能力", "应变能力", "处理能力",
                "规培", "轮转", "培训", "教学", "查房", "病例讨论", "教学查房",
                "白班", "急诊班", "手术班", "跟班", "带教",
                "掌握", "熟悉", "了解", "提高", "进步", "加强", "巩固",
                "麻醉技能", "临床技能", "操作技能", "专业技能", "基本技能",
                "麻醉操作", "临床操作", "技术操作", "基本操作",
                "值班"
            ],
            "人员角色": [
                "学员", "住院医师", "规培生", "实习生", "研究生", "进修生",
                "医师", "医生", "麻醉医师", "麻醉医生", "主治医师", "主任医师",
                "带教老师", "指导老师", "导师", "上级医师"
            ],
            "科室/场景": [
                "麻醉科", "手术室", "PACU", "恢复室", "ICU", "重症监护室",
                "急诊", "门诊", "病房", "导管室", "内镜中心", "无痛中心",
                "术前", "术中", "术后", "围术期", "围手术期",
                "值班", "胸科"
            ]
        }
        self.all_entities = []
        self.entity_to_category = {}
        for category, terms in self.relevance_entities.items():
            for term in terms:
                self.all_entities.append(term)
                self.entity_to_category[term] = category
        self.all_entities.sort(key=len, reverse=True)
        for entity in self.all_entities:
            jieba.add_word(entity, freq=1000)
        self.edu_keywords = [
            "技能", "能力", "态度", "学习", "培训", "值班", "轮转", "掌握",
            "熟悉", "了解", "提高", "进步", "加强", "巩固", "操作", "临床",
            "麻醉", "技术", "水平", "专业", "表现", "工作", "责任"
        ]
        self.person_keywords = [
            "学员", "住院医师", "规培生", "实习生", "医师", "医生", "麻醉"
        ]
        self.scene_keywords = [
            "麻醉科", "手术室", "PACU", "ICU", "急诊", "门诊", "病房",
            "术前", "术中", "术后", "围术期", "值班", "手术", "胸科"
        ]

    def _init_specificity_patterns_v3(self):
        self.strong_specificity = {
            "量化指标": [
                r"\d+\s*(mg|mcg|ml|mmHg|cm|度|min|分钟|小时|岁|kg|cmH2O|mmol/L|mEq/L|%)",
                r"第\s*\d+\s*次",
                r"\d+\s*点",
                r"\d+%",
                r"[一二三四五六七八九十百千万]+\s*(mg|ml|min|次|点)"
            ],
            "精确行为": [
                r"(穿刺|插管|操作|推注|给药|调整).{0,10}(角度|深度|速度|位置|方向|剂量|浓度|时间|手法)",
                r"(调整|改变|选择).{0,10}(为|成|到|至)",
                r"(使用|采用).{0,10}(方法|技术|方案|药物|设备|工具|入路)",
                r"(观察到|发现|注意到).{0,15}(了|有|出现|存在)"
            ],
            "情境锚定": [
                r"(术中|术后|诱导期|维持期|拔管前|穿刺时|给药后|翻身时|关腹时|缝合时)",
                r"(当|在).{0,10}(时|过程中|之后|之前|期间)",
                r"(本次|这次|今日|本例|该例|本周|本月)"
            ]
        }
        self.moderate_specificity = {
            "技能具体化": [
                r"(掌握|熟悉|了解).{0,8}(技能|操作|技术|能力|方法|流程|规范)",
                r"(能够|可以|独立).{0,8}(完成|进行|操作|实施|处理|管理)",
                r"(麻醉|临床|技术).{0,8}(操作|技能|能力|水平|流程|规范)"
            ],
            "场景具体化": [
                r"(在|于).{0,8}(手术室|麻醉科|PACU|ICU|急诊|门诊|病房).{0,10}(中|里|期间)",
                r"(负责|参与|主持|协助).{0,8}(手术|麻醉|值班|插管|穿刺)",
                r"(本周|本月|本次轮转|本阶段).{0,8}(完成|参与|负责|学习)"
            ],
            "对比具体化": [
                r"(较|比|相比).{0,8}(之前|以前|上次|初期|过去).{0,8}(进步|提高|改善|加强|熟练)",
                r"(从|由).{0,8}(不熟练|生疏|困难|不会).{0,8}(到|变为|发展为).{0,8}(熟练|掌握|独立|可以)"
            ]
        }
        self.weak_specificity = {
            "程度修饰": [
                r"(明显|显著|逐步|逐渐|进一步|有所|一定).{0,5}(进步|提高|改善|熟练|掌握)",
                r"(基本|初步|大体|大致).{0,5}(掌握|了解|熟悉|完成|独立)"
            ],
            "频次描述": [
                r"(多次|反复|经常|有时|偶尔).{0,5}(练习|操作|尝试|遇到|出现)",
                r"(共|累计|总计).{0,5}\d+.{0,5}(例|次|台|例次)"
            ]
        }
        self.pure_evaluative = [
            "很好", "不错", "优秀", "良好", "一般", "较差", "有待提高",
            "认真", "负责", "积极", "主动", "配合", "努力", "勤奋",
            "态度端正", "表现良好", "令人满意", "值得肯定", "表现不错",
            "工作认真", "态度认真", "学习认真", "表现优秀"
        ]
        self.SPECIFICITY_THRESHOLD = 2.5

    def _init_actionability_patterns_v3(self):
        self.direct_action = {
            "明确建议": [
                r"(建议|应|应当|需要|最好).{0,25}(调整|改变|增加|减少|使用|采用|练习|学习|注意|避免|尝试|控制|监测|观察|加强|完善|复习|查阅|观摩)",
                r"(下次|今后|以后|未来|后续).{0,15}(注意|尝试|使用|采用|调整|改进|加强|完善|控制|练习|复习|重点|关注)",
                r"(可|可以).{0,10}(考虑|尝试|改为|调整为|增加至|减少至|换用|加强|完善|补充|增加)"
            ],
            "具体方案": [
                r"(改为|调整为|增加至|减少至|换成|使用).{0,20}(mg|ml|mmHg|方法|技术|药物|方案|设备|策略|入路|浓度|速度)",
                r"(建议|推荐).{0,20}(方案|方法|策略|措施|步骤|计划|路径|流程|规范)",
                r"(参考|借鉴|依据|按照).{0,10}(指南|共识|文献|标准|规范|教材|图谱|视频)"
            ],
            "学习计划": [
                r"(建议|需要|应当).{0,20}(复习|学习|掌握|熟悉|了解|练习|培训|观摩|查阅|阅读|观看|总结|记录)",
                r"(目标|目的是).{0,15}(掌握|能够|学会|熟悉|了解|完成|独立|熟练|规范)",
                r"(重点|着重|加强|优先).{0,10}(学习|掌握|练习|提高|培养|关注|训练)"
            ]
        }
        self.indirect_action = {
            "进步空间": [
                r"(还有|仍有|尚|有待|需要).{0,10}(提高|改进|加强|完善|学习|练习|努力|进步)",
                r"(不够|不足|不太|不是很).{0,10}(熟练|熟悉|掌握|了解|清楚|规范|充分)"
            ],
            "问题指出": [
                r"(存在|出现|有).{0,10}(问题|不足|缺陷|错误|偏差|困难|风险|隐患)",
                r"(注意|警惕|防范|预防).{0,10}(风险|并发症|意外|错误|疏漏|不足)"
            ]
        }
        self.non_actionable = {
            "纯表扬": [
                r"(表现|工作|态度|能力|水平|技能).{0,10}(良好|优秀|不错|认真|负责|积极|较好|令人满意|值得肯定|有很大提高|进步明显)",
                r"(继续|保持).{0,10}(努力|良好|优秀|状态|水平|进步|作风|精神)"
            ],
            "纯祝福": [
                r"(希望|祝|愿|期待|相信).{0,15}(顺利|成功|进步|提高|更好|加油|成才|发展)",
                r"(争取|力图).{0,10}(更大进步|更好成绩|更高水平|更优秀)"
            ],
            "状态描述": [
                r"(目前|现在|现阶段).{0,10}(能够|可以|已经|基本).{0,10}(完成|掌握|独立|操作)",
                r"(总体|整体|综合来看).{0,10}(表现|情况|状态|水平).{0,10}(良好|不错|可以|满意)"
            ]
        }
        self.ACTIONABILITY_THRESHOLD = 3.0

    def assess_relevance(self, text):
        if not text or len(text.strip()) == 0:
            return False, {"reason": "Text is empty", "entities": [], "match_type": "none"}
        text_lower = text.lower()
        clinical_entities = []
        found_categories = set()
        for entity in self.all_entities:
            if entity in text or entity in text_lower:
                count = text.count(entity) + text_lower.count(entity)
                category = self.entity_to_category.get(entity, "Unknown")
                clinical_entities.append({"entity": entity, "category": category, "count": count})
                found_categories.add(category)
        seen = set()
        unique_entities = []
        for e in clinical_entities:
            if e["entity"] not in seen:
                seen.add(e["entity"])
                unique_entities.append(e)
        has_edu = any(kw in text for kw in self.edu_keywords)
        has_person = any(kw in text for kw in self.person_keywords)
        has_scene = any(kw in text for kw in self.scene_keywords)
        has_core = any(kw in text for kw in self.core_keywords)
        is_relevant = len(unique_entities) > 0 or has_edu or has_person or has_scene or has_core
        if len(unique_entities) > 0:
            match_type = "clinical_entity"
        elif has_core:
            match_type = "core_keyword"
        elif has_edu:
            match_type = "education_evaluation"
        elif has_person:
            match_type = "person_mention"
        elif has_scene:
            match_type = "scene_mention"
        else:
            match_type = "none"
        explanation = {
            "relevant": is_relevant,
            "entity_count": len(unique_entities),
            "categories": list(found_categories),
            "entities": unique_entities,
            "has_edu": has_edu,
            "has_person": has_person,
            "has_scene": has_scene,
            "has_core": has_core,
            "match_type": match_type,
            "reason": self._gen_relevance_reason(is_relevant, unique_entities, found_categories,
                                                  has_edu, has_person, has_scene, has_core)
        }
        return is_relevant, explanation

    def _gen_relevance_reason(self, is_relevant, entities, categories, has_edu, has_person, has_scene, has_core):
        if not is_relevant:
            return "No clinical entity, educational term, personnel, scenario, or core keyword identified"
        reasons = []
        if len(entities) > 0:
            category_summary = []
            for cat in categories:
                cat_entities = [e["entity"] for e in entities if e["category"] == cat]
                category_summary.append(f"{cat}({', '.join(cat_entities[:3])})")
            reasons.append(f"Clinical entities: {'; '.join(category_summary)}")
        if has_core:
            reasons.append("Core keyword match")
        if has_edu:
            reasons.append("Educational evaluation term")
        if has_person:
            reasons.append("Personnel mention")
        if has_scene:
            reasons.append("Scenario mention")
        return f"Matched dimensions: {'; '.join(reasons)}"

    def assess_specificity(self, text):
        if not text or len(text.strip()) == 0:
            return False, {"score": 0, "reason": "Text is empty"}
        score = 0.0
        evidence = []
        has_strong = False
        for category, patterns in self.strong_specificity.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    score += 3.0 * len(matches)
                    has_strong = True
                    evidence.extend([f"{category}[strong]: {m}" for m in matches[:2]])
        for category, patterns in self.moderate_specificity.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    score += 1.5 * len(matches)
                    evidence.extend([f"{category}[moderate]: {m}" for m in matches[:2]])
        weak_matches = 0
        for category, patterns in self.weak_specificity.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    weak_matches += len(matches)
                    evidence.extend([f"{category}[weak]: {m}" for m in matches[:1]])
        if weak_matches >= 2:
            score += 1.5
            evidence.append("Weak evidence combination +1.5")
        elif weak_matches == 1:
            score += 0.8
        pure_eval_count = sum(1 for term in self.pure_evaluative if term in text)
        word_count = len(jieba.lcut(text))
        if score < 2.0 and pure_eval_count > 0:
            has_skill = any(kw in text for kw in ["操作", "技能", "技术", "能力", "流程", "规范", "穿刺", "插管", "麻醉"])
            if not has_skill and word_count < 30:
                score -= 1.5
                evidence.append("Pure evaluative short text -1.5")
        is_specific = has_strong or score >= self.SPECIFICITY_THRESHOLD
        if is_specific:
            reason = f"Specificity score {score:.1f} (threshold {self.SPECIFICITY_THRESHOLD}), contains observable behaviors or quantitative indicators"
            if evidence:
                reason += f". Evidence: {evidence[:4]}"
        else:
            reason = f"Specificity score {score:.1f} (threshold {self.SPECIFICITY_THRESHOLD}), lacks observable behavioral details or quantitative indicators"
            if evidence:
                reason += f". Detected: {evidence[:3]}"
        explanation = {
            "specific": is_specific,
            "score": score,
            "threshold": self.SPECIFICITY_THRESHOLD,
            "has_strong_evidence": has_strong,
            "evidence": evidence,
            "reason": reason
        }
        return is_specific, explanation

    def assess_actionability(self, text):
        if not text or len(text.strip()) == 0:
            return False, {"score": 0, "reason": "Text is empty"}
        score = 0.0
        evidence = []
        has_direct_action = False
        for category, patterns in self.direct_action.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    score += 3.0 * len(matches)
                    has_direct_action = True
                    evidence.extend([f"{category}[direct]: {m}" for m in matches[:2]])
        for category, patterns in self.indirect_action.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    score += 1.0 * len(matches)
                    evidence.extend([f"{category}[indirect]: {m}" for m in matches[:2]])
        non_actionable_score = 0
        for category, patterns in self.non_actionable.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    non_actionable_score += 2.0 * len(matches)
                    evidence.extend([f"{category}[non-actionable]: {m}" for m in matches[:2]])
        if not has_direct_action:
            score -= non_actionable_score * 0.5
        else:
            score -= non_actionable_score * 0.2
        has_vague_suggestion = "建议" in text and not has_direct_action
        if has_vague_suggestion:
            score += 0.5
            evidence.append("Vague suggestion +0.5")
        is_actionable = has_direct_action or score >= self.ACTIONABILITY_THRESHOLD
        if is_actionable:
            reason = f"Actionability score {score:.1f} (threshold {self.ACTIONABILITY_THRESHOLD})"
            if has_direct_action:
                reason += ", contains explicit improvement directions or learning plans"
            else:
                reason += ", determined through multiple indirect signals"
            if evidence:
                reason += f". Evidence: {evidence[:4]}"
        else:
            reason = f"Actionability score {score:.1f} (threshold {self.ACTIONABILITY_THRESHOLD}), lacks explicit improvement directions or learning plans"
            if evidence:
                reason += f". Detected: {evidence[:3]}"
        explanation = {
            "actionable": is_actionable,
            "score": score,
            "threshold": self.ACTIONABILITY_THRESHOLD,
            "has_direct_action": has_direct_action,
            "evidence": evidence,
            "reason": reason
        }
        return is_actionable, explanation

    def diagnose(self, text):
        rel, rel_exp = self.assess_relevance(text)
        spec, spec_exp = self.assess_specificity(text)
        act, act_exp = self.assess_actionability(text)
        if not rel:
            utility = "Irrelevant"
            stage1_pass = False
        elif spec and act:
            utility = "Effective"
            stage1_pass = True
        elif spec or act:
            utility = "Moderate"
            stage1_pass = True
        else:
            utility = "Ineffective"
            stage1_pass = False
        report = {
            "stage": "RSA Diagnostic Baseline v3.0 (Extended)",
            "utility": utility,
            "passed_baseline": stage1_pass,
            "dimensions": {
                "relevance": {"passed": rel, **rel_exp},
                "specificity": {"passed": spec, **spec_exp},
                "actionability": {"passed": act, **act_exp}
            },
            "missing_dimensions": []
        }
        if not rel: report["missing_dimensions"].append("relevance")
        if not spec: report["missing_dimensions"].append("specificity")
        if not act: report["missing_dimensions"].append("actionability")
        return report


# ==================== Helper: NLP feature extraction (for batch ML prediction) ====================
def extract_nlp_features_for_batch(texts, prof_words, high_utility_keywords):
    """
    Extract 17 NLP statistical features, consistent with training.
    """
    positive = ["好", "优秀", "出色", "正确", "成功", "顺利"]
    negative = ["差", "错误", "失败", "不足", "问题", "缺点"]
    punct = ["。", "，", "；", "：", "！", "？"]
    features = []
    for t in texts:
        words = t.split()
        if not words:
            features.append([0.0] * 17)
            continue
        length = len(t)
        word_count = len(words)
        unique = len(set(words))
        type_ratio = unique / word_count if word_count else 0
        prof_cnt = sum(1 for w in words if w in prof_words)
        prof_ratio = prof_cnt / word_count
        pos_cnt = sum(1 for w in words if w in positive)
        neg_cnt = sum(1 for w in words if w in negative)
        senti = pos_cnt - neg_cnt
        punct_cnt = sum(1 for ch in t if ch in punct)
        digit_cnt = sum(1 for ch in t if ch.isdigit())
        kw_feats = []
        for kw_list in high_utility_keywords.values():
            cnt = sum(1 for w in words if any(k in w for k in kw_list))
            kw_feats.append(cnt)
        feats = [length, word_count, unique, type_ratio, prof_cnt, prof_ratio,
                 pos_cnt, neg_cnt, senti, punct_cnt, digit_cnt] + kw_feats
        features.append(feats)
    return np.array(features, dtype=np.float64)


# ==================== Streamlit App ====================
def main():
    st.set_page_config(page_title="Anesthesia Feedback Utility Evaluation System v4.0", layout="wide")
    st.title("🏥 Anesthesia Feedback Utility Evaluation System")
    st.caption("v4.0 — RSA-based three‑dimensional diagnostic framework  |  Relevance · Specificity · Actionability")

    # Initialize RSA
    rsa = RSADiagnosticBaseline()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        - **Single analysis**: Enter feedback in the main panel and click "Analyze"
        - **Batch testing**: Upload an Excel file with a "反馈文本" column for bulk evaluation and export
        """)

        st.divider()
        st.header("⚙️ Batch Test Settings")
        
        analysis_mode = st.radio(
            "Select analysis engine",
            ["RSA rule diagnosis (no model)", "ML model prediction (upload .pkl)"],
            index=0
        )
        
        model_pipeline = None
        if analysis_mode == "ML model prediction (upload .pkl)":
            uploaded_model = st.file_uploader(
                "Upload trained pipeline (.pkl)",
                type=["pkl"],
                help="Must contain model, scaler, and vectorizer"
            )
            if uploaded_model is not None:
                try:
                    model_pipeline = pickle.load(uploaded_model)
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Model loading failed: {e}")
                    model_pipeline = None
            else:
                st.info("Please upload a model file to enable ML prediction")
                model_pipeline = None
        
        st.divider()
        uploaded_file = st.file_uploader(
            "📂 Upload Excel file for batch testing",
            type=["xlsx", "xls"],
            help="File must contain a column named '反馈文本'"
        )
        
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.success("History cleared")

    # Initialize history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # ---- Main tabs ----
    tab1, tab2 = st.tabs(["📝 Single Analysis", "📊 Batch Testing"])

    # ---------- Tab 1: Single Analysis ----------
    with tab1:
        col_input, col_result = st.columns([1, 1.5])
        with col_input:
            st.subheader("📝 Feedback Text Input")
            input_text = st.text_area(
                "Enter anesthesia teaching feedback",
                height=200,
                placeholder="e.g., The resident has shown improved thoracic anesthesia skills and independent on-call ability, but remains hesitant in managing sudden hemodynamic changes."
            )
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

            st.caption("Quick sample tags:")
            sample_tags = ["胸科麻醉", "独立值班", "血流动力学", "困难气道", "术后镇痛"]
            sample_cols = st.columns(len(sample_tags))
            for i, tag in enumerate(sample_tags):
                if sample_cols[i].button(tag, key=f"tag_{i}"):
                    input_text = tag

        with col_result:
            st.subheader("📊 Evaluation Result")
            if analyze_btn and input_text.strip():
                report = rsa.diagnose(input_text)
                utility = report['utility']
                dims = report['dimensions']

                score = 0.0
                if dims['relevance']['passed']: score += 0.4
                if dims['specificity']['passed']: score += 0.3
                if dims['actionability']['passed']: score += 0.3
                score = round(score, 3)

                if utility == "Effective":
                    icon = "✅"
                elif utility == "Moderate":
                    icon = "⚠️"
                elif utility == "Ineffective":
                    icon = "❌"
                else:
                    icon = "⏹️"

                st.markdown(f"### {icon} Utility Level: **{utility}**")
                st.metric("Composite Score", f"{score:.3f}", delta=None)

                with st.expander("🔍 RSA Three‑Dimension Diagnosis Details", expanded=True):
                    rel = dims['relevance']
                    st.markdown(f"**Relevance**  {'✅ Passed' if rel['passed'] else '❌ Failed'}")
                    st.progress(0.8 if rel['passed'] else 0.2)
                    st.caption(f"Evidence: {rel.get('reason', 'None')}")
                    if rel.get('entities'):
                        st.write("Matched entities:", [e['entity'] for e in rel['entities'][:5]])

                    spec = dims['specificity']
                    st.markdown(f"**Specificity**  {'✅ Passed' if spec['passed'] else '❌ Failed'}")
                    st.progress(0.8 if spec['passed'] else 0.2)
                    st.caption(f"Score: {spec.get('score', 0):.1f}, {spec.get('reason', '')}")

                    act = dims['actionability']
                    st.markdown(f"**Actionability**  {'✅ Passed' if act['passed'] else '❌ Failed'}")
                    st.progress(0.8 if act['passed'] else 0.2)
                    st.caption(f"Score: {act.get('score', 0):.1f}, {act.get('reason', '')}")

                st.subheader("💡 Improvement Suggestions")
                suggestions = []
                if not dims['relevance']['passed']:
                    suggestions.append("• Include specific clinical procedures, drugs, or monitoring indicators to enhance relevance.")
                if not dims['specificity']['passed']:
                    suggestions.append("• Provide more detailed behavioral descriptions or quantitative data (e.g., dose, timing, location) to improve specificity.")
                if not dims['actionability']['passed']:
                    suggestions.append("• Clearly state improvement directions or concrete learning plans (e.g., 'Practice fiberoptic intubation') to increase actionability.")
                if not suggestions:
                    suggestions.append("🎉 This feedback performs well across all three dimensions. Keep it up!")
                for s in suggestions:
                    st.write(s)

                st.session_state.history.append({
                    "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Text": input_text[:50] + ("..." if len(input_text)>50 else ""),
                    "Utility": utility,
                    "Score": score
                })
            else:
                st.info("Enter feedback text and click 'Analyze' to see results.")

    # ---------- Tab 2: Batch Testing ----------
    with tab2:
        if uploaded_file is None:
            st.info("Please upload an Excel file in the sidebar to start batch testing.")
        else:
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"File reading error: {e}")
                st.stop()
            
            if '反馈文本' not in df.columns:
                st.error("❌ Excel file must contain a column named '反馈文本'.")
                st.stop()
            
            original_len = len(df)
            df = df.dropna(subset=['反馈文本'])
            df = df[df['反馈文本'].astype(str).str.strip() != '']
            st.info(f"📊 Loaded {len(df)} valid feedback entries (original {original_len}, removed {original_len - len(df)} blanks)")
            
            if len(df) == 0:
                st.warning("No valid feedback text found.")
                st.stop()
            
            if st.button("🚀 Start Batch Analysis", type="primary"):
                with st.spinner("Analyzing, please wait..."):
                    results = []
                    if analysis_mode == "RSA rule diagnosis (no model)":
                        for idx, row in df.iterrows():
                            text = row['反馈文本']
                            report = rsa.diagnose(str(text))
                            utility = report['utility']
                            dims = report['dimensions']
                            score = 0.0
                            if dims['relevance']['passed']: score += 0.4
                            if dims['specificity']['passed']: score += 0.3
                            if dims['actionability']['passed']: score += 0.3
                            results.append({
                                'Feedback Text': text,
                                'Utility Level': utility,
                                'Composite Score': round(score, 3),
                                'Relevance Passed': dims['relevance']['passed'],
                                'Specificity Score': dims['specificity']['score'],
                                'Actionability Score': dims['actionability']['score']
                            })
                    else:  # ML mode
                        if model_pipeline is None:
                            st.error("❌ Please upload a valid model file first.")
                            st.stop()
                        model = model_pipeline.get('model')
                        scaler = model_pipeline.get('scaler')
                        vectorizer = model_pipeline.get('vectorizer')
                        if model is None or scaler is None or vectorizer is None:
                            st.error("❌ Pipeline missing required components (model/scaler/vectorizer)")
                            st.stop()
                        
                        def simple_preprocess(t):
                            if pd.isna(t) or not isinstance(t, str) or len(t.strip())==0:
                                return ""
                            t = re.sub(r'[^\w\u4e00-\u9fff。，；：！？、]', ' ', t)
                            t = re.sub(r'\s+', ' ', t).strip()
                            if len(t)==0:
                                return ""
                            words = jieba.lcut(t, cut_all=False)
                            stopwords = set(['的','了','在','是','我','有','和','就','不','人','都','一','一个','上','也','很','到','说','要','去','你','会',
                                            '着','没有','看','好','自己','这','那','他','她','它','而','但','与','之','或','且','及','并','对于','关于',
                                            '对','从','向','以','为','于','把','被','让','给','使','可以','可能','能够','应该','必须','需要','要求','一定'])
                            words = [w for w in words if len(w)>1 and w not in stopwords]
                            return ' '.join(words)
                        
                        processed_texts = [simple_preprocess(t) for t in df['反馈文本']]
                        
                        prof_words = []
                        try:
                            with open(os.path.join(SAVE_DIR, "麻醉专业词库.txt"), 'r', encoding='utf-8') as f:
                                prof_words = [line.strip() for line in f]
                        except:
                            st.warning("Professional vocabulary file not found; NLP professional term counts will be zero, may affect accuracy.")
                            prof_words = []
                        
                        high_utility_keywords = {
                            "行为中心": ["行为", "操作", "表现", "动作", "医师", "医生", "学员", "住院医师"],
                            "详细信息": ["详细", "具体", "明确", "清晰", "描述", "记录", "记载", "说明"],
                            "负面反馈": ["不足", "改进", "错误", "问题", "缺点", "欠缺", "需要提高", "不够", "未", "缺乏", "差", "不好"],
                            "专业性": ["专业", "沟通", "交流", "解释", "告知", "说明", "计划", "准备", "能力", "技能"],
                            "针对性": ["针对", "特定", "具体", "相关", "对应", "针对性地", "专门", "特别"],
                            "改进方向": ["改进", "提高", "加强", "完善", "优化", "建议", "应该", "需要", "要", "可以", "改进方向"]
                        }
                        
                        X_tfidf = vectorizer.transform(processed_texts).toarray()
                        X_nlp = extract_nlp_features_for_batch(processed_texts, prof_words, high_utility_keywords)
                        X = np.hstack([X_tfidf, X_nlp])
                        
                        try:
                            X_scaled = scaler.transform(X)
                        except Exception as e:
                            st.error(f"❌ Feature standardization failed: {e}. Ensure model expects 117 features.")
                            st.stop()
                        
                        y_pred = model.predict(X_scaled)
                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(X_scaled)
                            if probs.shape[1] == 2:
                                prob_pos = probs[:, 1]
                            else:
                                prob_pos = np.max(probs, axis=1)
                        else:
                            prob_pos = [0.5] * len(y_pred)
                        
                        label_map = model_pipeline.get('label_map', {0: 'Irrelevant', 1: 'Ineffective', 2: 'Moderate', 3: 'Effective'})
                        for i, text in enumerate(df['反馈文本']):
                            pred_label = y_pred[i]
                            utility = label_map.get(pred_label, str(pred_label))
                            prob = prob_pos[i] if hasattr(model, 'predict_proba') else 0.5
                            results.append({
                                'Feedback Text': text,
                                'Predicted Utility': utility,
                                'Confidence (High Utility Prob)': round(prob, 3)
                            })
                    
                    result_df = pd.DataFrame(results)
                    
                    st.subheader("📈 Batch Analysis Results")
                    
                    # Plot distribution if utility column exists
                    if 'Utility Level' in result_df.columns or 'Predicted Utility' in result_df.columns:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Total Samples", len(result_df))
                            label_col = 'Utility Level' if 'Utility Level' in result_df.columns else 'Predicted Utility'
                            counts = result_df[label_col].value_counts()
                            st.write("Distribution:")
                            st.dataframe(counts.reset_index().rename(columns={'index': 'Level', label_col: 'Count'}))
                        with col2:
                            # ---- Plot with English titles ----
                            fig, ax = plt.subplots()
                            # Ensure all categories are present for consistent colors
                            categories = ['Effective', 'Moderate', 'Ineffective', 'Irrelevant']
                            # Reindex counts to include missing categories with 0
                            counts = counts.reindex(categories, fill_value=0)
                            colors = ['#2ECC71', '#F39C12', '#E74C3C', '#95A5A6']
                            bars = ax.bar(counts.index, counts.values, color=colors)
                            ax.set_xlabel('Utility Level', fontsize=12)
                            ax.set_ylabel('Count', fontsize=12)
                            ax.set_title('Distribution of Feedback Utility', fontsize=14)
                            # Add value labels on bars
                            for bar in bars:
                                height = bar.get_height()
                                if height > 0:
                                    ax.annotate(f'{int(height)}',
                                                xy=(bar.get_x() + bar.get_width()/2, height),
                                                xytext=(0, 3),
                                                textcoords="offset points",
                                                ha='center', va='bottom')
                            st.pyplot(fig)
                    
                    st.subheader("📋 Detailed Results (downloadable)")
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="feedback_batch_results.csv",
                        mime="text/csv"
                    )
                    
                    st.success(f"✅ Batch analysis complete! Processed {len(result_df)} entries.")

    # ---- Global history section ----
    st.divider()
    st.subheader("📜 History (Single Analysis)")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        st.caption(f"Total {len(df_hist)} records, average score: {df_hist['Score'].mean():.3f}")
    else:
        st.caption("No single analysis records yet.")

    st.divider()
    st.caption("Anesthesia Feedback Utility Evaluation System v4.0 - Streamlit Edition  |  RSA Three‑Dimension Framework")

if __name__ == "__main__":
    main()