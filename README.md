# sim-simple
simple version of autonomous driving simulation engine.

## Setup
### 1. start 'Conda Prompt'
### 2. Create environment from `environment.yml`
'''bash
conda env creat -f environment.yml
conda activate sim


# Description

## 1. 자율주행과 시뮬레이션

자율주행은 환경을 센서로 관측하고(Perception), 그 결과로 판단/계획하고(Planning), 차량을 물리적으로 제어한 뒤(Control/Dynamics), 다시 환경이 바뀌는 폐루프(Closed-loop) 시스템이다.

Autonomous Driving (closed-loop)
Environment → Sensors(Camera/LiDAR/…) → Perception → Path Planning → Control(+Dynamics) → Environment

Human Driving (개념적 비교)
Environment → Human(인지/판단/행동) → Environment

### 구별 단계
 - Open-Loop Simulation/Closed-Loop Simulation
 - Object-Level/Sensor-Level
    Open-Loop Object-Level이 제일 쉬움
    해당 프로젝트는 성능은 매우 낮지만 Closed-Loop, Sensor-Level임

## 2. 시뮬레이션과 현실 차이점

Sim2Real Gap: 시뮬레이터가 아무리 좋아도 현실의 “빛/그림자/반사(photorealism), 센서 노이즈, 재질, 날씨, 동적 에이전트” 등을 완벽히 재현하기는 어렵다.

Digital Twin: 현실과 최대한 닮게 만들려면, 도로/표지/차량/보행자 등 자산(assets)의 종류·크기·재질·배치가 중요하고, 센서 모델까지 포함한 “물리 기반” 재현이 핵심이 된다. 
 - NVIDIA는 물리 기반 센서 시뮬레이션/디지털 트윈 워크플로우를 전면에 둠.

## 3. 현업에서 시뮬레이션을 쓰는 이유 (검증 + 데이터)

검증(Validation/Verification): 현실과 유사할수록 각 모듈의 성능/안전성을 더 신뢰성 있게 평가 가능. Waymo는 시뮬레이션을 “전체 시스템 평가 도구”로 강하게 활용한다고 공개함.

학습용 데이터(Synthetic Data): 실차 데이터가 부족한 코너 케이스를 대량 생성해 학습/회귀테스트에 투입. NVIDIA도 Omniverse/Isaac Sim 기반 합성데이터 파이프라인을 대표 유스케이스로 제시.

Ground Truth 관점: 실차에서는 “가볍고 검증된 모델로 inference만” 하고, 학습/튜닝은 시뮬·오프라인 파이프라인에서 데이터로 간접 기여(네가 적은 방향이 실제 실무 흐름과 잘 맞음).

## 4. 시뮬레이션을 굴리기 위한 4개 모듈 (기능 · 입력 · 출력 · 확장)

이 프로젝트는 아래 4개가 최소로 맞물리면 “돌아가는 자율주행 시뮬”이 된다.

1) Map / World (환경·지도 모듈)

역할(기능)
    도로/차선/경계/신호/정적 객체 등 월드의 기준(ground truth) 형상 제공
    전역좌표계(예: lat/lon, UTM 등) ↔ 로컬좌표계(ego/ENU/BEV) 변환의 기준 제공

입력(Input)
    HD Map / OSM / OpenDRIVE 등 지도 데이터
    시뮬 시작 포즈(ego 초기 위치/방향), 좌표계 기준점(ref lat/lon 등)

출력(Output)
    시뮬 월드 objects: polylines / polygons / landmarks
    예: “차선 중심선 샘플”, “근방 N미터 객체”, “충돌 체크용 도형”

확장 방향(Extension)
    (정적→동적) 신호등 상태 머신, 공사 구간, 가변 차로
    (정밀도) OpenDRIVE 파라메트릭 차선, 곡률/캠버/고도 반영
    (성능) 공간 인덱싱(R-tree/KD-tree), 타일링/스트리밍 로딩

2) Perception / Sensors (센서 + 인지 모듈)

역할(기능)
    월드에서 센서 관측치 생성(시뮬) 또는 실차 로그 입력(리플레이)
    관측치를 해석해 인지 결과(world model) 생성
    예: 차선/장애물/자차 자세/가용 공간(occupancy) 등

입력(Input)
    월드 상태(정적/동적 객체), ego pose/state, 센서 파라미터(내·외부, FOV, 노이즈 등)     
    
출력(Output)
    “인지 결과” 표준 형태 예시:   
     - ego_state, detected_objects, lane_polylines, free_space/occupancy, traffic_light_state …

확장 방향(Extension)
    (센서 고도화) 카메라·라이다·레이다 물리 기반 렌더링/노이즈/가림(occlusion)
    (인지 고도화) multi-object tracking, sensor fusion, uncertainty(공분산) 출력
    (현업 연결) NVIDIA는 Omniverse/센서 RTX API로 고충실도 센서 시뮬레이션을 강조

3) Path Planning (판단/경로계획 모듈)
역할(기능)
    인지 결과 + 목표(waypoint, route)로부터 주행 의사결정 + 기준 경로(reference path/trajectory) 생성
    Behavior(차선 변경/정지/양보) + Motion Planning(경로/속도 프로파일)

입력(Input)
    인지 world model, 지도(차선 토폴로지), 목표/제약(속도 제한, 충돌 회피, comfort)

출력(Output)
    reference_path (centerline/trajectory points)
    speed_profile 또는 시간 파라메트릭 trajectory (x(t), y(t), v(t), a(t))

확장 방향(Upgrade path)
    pure pursuit용 기준선 → (중급) lattice / sampling / optimization
    멀티에이전트 상호작용, 예측 기반 MPC, 위험도(cost) 학습
    closed-loop 시뮬에서 “실패 케이스 자동 생성/회귀”가 중요(웨이모 시뮬레이션)

4) Control + Dynamics (제어 + 차량 동역학 모듈)

역할(기능)
    계획된 trajectory를 실제 차량 입력(조향/가감속)으로 추종
    dynamics는 “차량이 입력에 어떻게 반응하는지”를 시뮬레이션(모델)

입력(Input)
    reference trajectory/path, ego state(속도/헤딩/위치), 차량 파라미터(wheelbase 등)

출력(Output)
    제어 입력: steer, throttle, brake (또는 accel, steer_rate)
    다음 frame의 ego state (dynamics step 결과)

확장 방향(Extention)
    (추종기) PID / Pure Pursuit → Stanley → MPC
    (동역학) kinematic → dynamic bicycle(타이어/슬립) → full vehicle model
    (현업) sim2real에서 특히 타이어/제동/노면 마찰이 오차의 큰 원인이라, 검증 목적이면 여기 현실성이 체감됨

## 5. 현업 케이스
Tesla :상대적으로 “통합형(end-to-end)”에 더 가까운 흐름
    공개적으로 FSD v12 은 도심 주행 스택을 단일 신경망으로 통합했다고 알려져 있음.
    즉, 전통적 “Perception→Planning→Control” 경계를 엄격히 나누기보다 학습된 정책이 여러 단계를 흡수하는 접근으로 이해하면 좋음.

NVIDIA (시뮬레이션/합성데이터/디지털트윈 “플랫폼”)
    Omniverse/Isaac Sim 중심으로 디지털 트윈 + 물리 기반 센서 시뮬 + 합성데이터 생성 워크플로우를 제공(차량 SW 스택 그 자체라기보다 “개발 인프라/툴체인”에 강점).

현대차/모셔널 (실차 개발 + 폐쇄코스 검증 + 상용 로보택시 지향)
    Hyundai–Motional 협력으로 IONIQ 5 기반 로보택시 개발/검증 및 시험을 공개적으로 진행.
    프레임으로 보면 “World/Map + 실차 센서/인지 + 계획/제어” 전통 모듈형을 강하게 운영하되, 시뮬/시험장을 함께 쓰는 형태로 이해 가능.

Waymo (모듈형 스택 + 대규모 closed-loop 시뮬 평가)
    Waymo는 “SimulationCity” 같은 시뮬 도구로 Waymo Driver 전체를 평가한다고 공개했고, 연구 측면에선 Waymax 같은 시나리오 기반 시뮬 라이브러리도 공개함.

