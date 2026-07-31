# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SHWX 数据集 25 个类别的 CLIP 文本提示词。

每个类别提供多个 prompt 模板（包含完整型号名、北约代号、机翼构型、
发动机布局、遥感视角等），构建语义相似度矩阵时对同一类别的所有
prompt 编码结果取平均，以增强 CLIP 对类别语义的稳定理解。

提示词编写原则：
- 使用完整型号名而非缩写（CLIP 对缩写理解不稳定）。
- 描述可见外观特征：机翼构型、发动机数量与布局、特殊雷达/天线特征。
- 统一使用遥感/俯视视角描述，与检测任务视角一致。
- 对无法准确解释的代号类（如 FSC），注明使用通用描述。

类别索引与 SHWX 数据集 data.yaml 中的 `names` 保持一致。
"""

# ruff: noqa: E501  -- 提示词为自然语言文本，不应按代码行长 120 强制换行

from __future__ import annotations

# 类别索引 → 提示词列表
SHWX_CLASS_PROMPTS: dict[int, list[str]] = {
    # ------------------------------------------------------------------
    # 舰船类 (0-3)：遥感俯视视角下的船体形态
    # ------------------------------------------------------------------
    0: [  # HM —— 航空母舰 (Aircraft Carrier)
        "an aircraft carrier with a huge flat flight deck and an island superstructure on one side, viewed from above in satellite imagery",
        "a large military warship with an angled runway painted on its flight deck and aircraft parked on deck, overhead aerial view",
        "a nuclear-powered aircraft carrier with rectangular flight deck and angled landing runway, top-down remote sensing image",
    ],
    1: [  # LQS —— 两栖攻击舰 (Amphibious Assault Ship)
        "an amphibious assault ship with a wide flat straight flight deck and helicopters, overhead satellite view",
        "a landing helicopter dock amphibious warship resembling a small aircraft carrier but with a straight flight deck, aerial remote sensing image",
        "a large amphibious warfare ship with wide deck and helipads, smaller than an aircraft carrier, top-down view",
    ],
    2: [  # QHS —— 驱逐舰/护卫舰 (Destroyers & Frigates)
        "a destroyer warship with an elongated narrow hull and superstructure in the middle, overhead satellite view",
        "a naval destroyer or frigate with missile launchers and a tall mast, slender hull, top-down remote sensing image",
        "an escort warship with a narrow beam and central bridge superstructure, much smaller than a carrier, aerial view",
    ],
    3: [  # MS —— 民用船舶 (Merchant / Civilian Ships)
        "a merchant cargo ship with rows of shipping containers stacked on its deck, overhead satellite view",
        "a civilian cargo vessel with a long hull and blocky containers or holds, remote sensing image",
        "a commercial transport ship without military superstructure, elongated hull, aerial top-down view",
    ],
    # ------------------------------------------------------------------
    # 军用飞机类 (4-23)
    # ------------------------------------------------------------------
    4: [  # A1_SU-35 —— 苏-35 战斗机 (Sukhoi Su-35, NATO: Flanker-E)
        "a Sukhoi Su-35 fighter jet with twin engines, twin tail fins and large swept-back delta wings, overhead satellite view",
        "a Russian Su-35 Flanker fighter with a single seat canopy and side-by-side twin engine exhausts, top-down aerial image",
    ],
    5: [  # A2_C-130 —— C-130 运输机 (Lockheed C-130 Hercules)
        "a Lockheed C-130 Hercules transport aircraft with four propeller engines and high-mounted wings, overhead satellite view",
        "a four turboprop military cargo plane with straight high wing and fuselage-mounted landing gear pods, top-down aerial image",
        "a C-130 tactical transport with a boxy fuselage and four propeller nacelles, remote sensing view",
    ],
    6: [  # A3_C-17 —— C-17 运输机 (Boeing C-17 Globemaster III)
        "a Boeing C-17 Globemaster III transport aircraft with four jet engines, high wing and T-tail, overhead satellite view",
        "a large military cargo jet with a blunt fuselage and four underwing engine pods, top-down aerial image",
        "a C-17 strategic transport with swept high wing and T-shaped tail, larger than C-130, remote sensing view",
    ],
    7: [  # A4_C-5 —— C-5 运输机 (Lockheed C-5 Galaxy)
        "a Lockheed C-5 Galaxy strategic transport aircraft with four jet engines and a huge high wing, the largest US military cargo plane, overhead satellite view",
        "a very large military heavy-lift transport with four engines and a massive upswept fuselage, top-down aerial image",
    ],
    8: [  # A5_F-16 —— F-16 战斗机 (General Dynamics F-16 Fighting Falcon)
        "a General Dynamics F-16 Fighting Falcon fighter jet with a single engine, single tail fin and a side-mounted air intake, overhead satellite view",
        "a single-engine F-16 fighter with a bubble canopy and small delta wing with leading edge extensions, top-down aerial image",
    ],
    9: [  # A6_TU-160 —— 图-160 轰炸机 (Tupolev Tu-160, NATO: Blackjack)
        "a Tupolev Tu-160 Blackjack strategic bomber with variable-sweep wings and twin tail fins, the largest supersonic bomber, overhead satellite view",
        "a huge white Russian strategic bomber with swept-back variable geometry wings and four paired engines, top-down aerial image",
    ],
    10: [  # A7_E-3 —— E-3 预警机 (Boeing E-3 Sentry AWACS)
        "a Boeing E-3 Sentry AWACS early warning aircraft with a large circular rotodome radar dish on top of the fuselage, overhead satellite view",
        "an E-3 airborne warning and control aircraft converted from a Boeing 707 with a mushroom-shaped radar dome above the fuselage, top-down aerial image",
    ],
    11: [  # A8_B-52 —— B-52 轰炸机 (Boeing B-52 Stratofortress)
        "a Boeing B-52 Stratofortress long-range bomber with eight engines in four twin pods under high wings and a long slender fuselage, overhead satellite view",
        "a large strategic bomber with eight jet engines grouped in pairs and swept wings, top-down aerial image",
    ],
    12: [  # A9_P-3C —— P-3C 反潜巡逻机 (Lockheed P-3C Orion)
        "a Lockheed P-3C Orion anti-submarine patrol aircraft with four turboprop engines and a magnetic anomaly detector tail stinger, overhead satellite view",
        "a four propeller maritime patrol aircraft with a long tail boom and weapons bay, top-down aerial image",
    ],
    13: [  # A10_B-1B —— B-1B 轰炸机 (Rockwell B-1B Lancer)
        "a Rockwell B-1B Lancer strategic bomber with variable-sweep wings and a blended wing-body design, overhead satellite view",
        "a swing-wing supersonic bomber with four small engines and a tapering nose, top-down aerial image",
    ],
    14: [  # A11_E-8 —— E-8 战场监视机 (Northrop Grumman E-8 Joint STARS)
        "a Northrop Grumman E-8 Joint STARS battlefield surveillance aircraft converted from a Boeing 707 with a canoe-shaped radome under the forward fuselage, overhead satellite view",
        "an E-8 ground surveillance aircraft with a long white ventral radar fairing and four engines under swept wings, top-down aerial image",
    ],
    15: [  # A12_TU-22 —— 图-22M 轰炸机 (Tupolev Tu-22M, NATO: Backfire)
        "a Tupolev Tu-22M Backfire bomber with variable-sweep wings and two large engine nacelles, overhead satellite view",
        "a Russian swing-wing supersonic bomber with twin engines mounted in the fuselage and swept back wings, top-down aerial image",
    ],
    16: [  # A13_F-15 —— F-15 战斗机 (McDonnell Douglas F-15 Eagle)
        "a McDonnell Douglas F-15 Eagle air superiority fighter with twin engines and twin tail fins, overhead satellite view",
        "a twin-engine F-15 fighter with a flat wide fuselage and two canted vertical tails, top-down aerial image",
    ],
    17: [  # A14_KC-135 —— KC-135 加油机 (Boeing KC-135 Stratotanker)
        "a Boeing KC-135 Stratotanker aerial refueling aircraft with four engines under swept wings and a boom, overhead satellite view",
        "a narrow-bodied tanker aircraft converted from a Boeing 707 with four underwing jet engines, top-down aerial image",
    ],
    18: [  # A15_F-22 —— F-22 战斗机 (Lockheed Martin F-22 Raptor)
        "a Lockheed Martin F-22 Raptor stealth fighter with diamond-shaped wings and canted twin tails, overhead satellite view",
        "a stealth fifth-generation fighter with a flat angular airframe and twin engines, top-down aerial image",
    ],
    19: [  # A16_FA-18 —— F/A-18 战斗机 (Boeing F/A-18 Super Hornet)
        "a Boeing F/A-18 Super Hornet carrier-based fighter with twin engines and twin tail fins, overhead satellite view",
        "a twin-engine carrier fighter with a blunt nose and large wing leading edge extensions, top-down aerial image",
    ],
    20: [  # A17_TU-95 —— 图-95 轰炸机 (Tupolev Tu-95, NATO: Bear)
        "a Tupolev Tu-95 Bear strategic bomber with four huge contra-rotating propeller engines and swept wings, overhead satellite view",
        "a large Russian turboprop bomber with eight-bladed contra-rotating propellers on four nacelles, top-down aerial image",
    ],
    21: [  # A18_KC-10 —— KC-10 加油机 (McDonnell Douglas KC-10 Extender)
        "a McDonnell Douglas KC-10 Extender aerial refueling aircraft converted from a DC-10 with three engines and a refueling boom, overhead satellite view",
        "a wide-body tanker aircraft with two engines under the wings and one at the tail base, top-down aerial image",
    ],
    22: [  # A19_SU-34 —— 苏-34 战斗机 (Sukhoi Su-34, NATO: Fullback)
        "a Sukhoi Su-34 Fullback fighter-bomber with a flattened duckbill nose, twin engines and canards, overhead satellite view",
        "a Russian strike aircraft with side-by-side cockpit seats and a wide flattened forward fuselage, top-down aerial image",
    ],
    23: [  # A20_SU-24 —— 苏-24 战斗轰炸机 (Sukhoi Su-24, NATO: Fencer)
        "a Sukhoi Su-24 Fencer fighter-bomber with variable-sweep wings and twin tail fins, overhead satellite view",
        "a Russian swing-wing attack aircraft with a long pointed nose and twin engines, top-down aerial image",
    ],
    # ------------------------------------------------------------------
    # 车辆类 (24)
    # ------------------------------------------------------------------
    24: [  # FSC —— 导弹发射车 (Missile Launch Vehicle / Transporter-Erecter-Launcher, TEL)
        # 注：类别有严格限制，仅指导弹发射车（带发射筒/发射架的军用车辆），非普通军用车辆。
        # 具体导弹型号未知，使用通用 TEL 描述（轮式底盘 + 圆柱形发射筒/箱式发射架）。
        "a missile launch vehicle with cylindrical launch tubes mounted on the rear chassis, overhead satellite view",
        "a transporter-erector-launcher military truck with launch canisters on its flatbed, top-down aerial image",
        "a wheeled missile transporter erector launcher vehicle with boxy launcher racks, remote sensing image",
    ],
}

# 类别索引 → 人类可读类别名称（用于日志与矩阵验证输出）
SHWX_CLASS_NAMES: dict[int, str] = {
    0: "HM",
    1: "LQS",
    2: "QHS",
    3: "MS",
    4: "A1_SU-35",
    5: "A2_C-130",
    6: "A3_C-17",
    7: "A4_C-5",
    8: "A5_F-16",
    9: "A6_TU-160",
    10: "A7_E-3",
    11: "A8_B-52",
    12: "A9_P-3C",
    13: "A10_B-1B",
    14: "A11_E-8",
    15: "A12_TU-22",
    16: "A13_F-15",
    17: "A14_KC-135",
    18: "A15_F-22",
    19: "A16_FA-18",
    20: "A17_TU-95",
    21: "A18_KC-10",
    22: "A19_SU-34",
    23: "A20_SU-24",
    24: "FSC",
}
