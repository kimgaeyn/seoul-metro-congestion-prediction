# ==============================================================================
# make_catchment_weights.py
# ------------------------------------------------------------------------------
# 역세권(반경 R) catchment 가중치 생성 — pipeline.py 의 입력(station_catchment_weights.csv).
#
# 방법: 지하철역 좌표를 행정동 경계에 버퍼-교차(area-weighted)하여
#       역마다 '겹치는 동 + 면적비율' 을 산출. 단일동 할당의 경계 역 오배정 문제를 제거.
#
# 산출물(station_catchment_weights.csv)을 이미 제공하므로 보통 재실행 불필요.
# 반경(R)을 바꾸거나 경계/좌표를 갱신할 때만 실행.
#
# 필요 입력 (서울 열린데이터광장 / 서울교통공사 공개데이터):
#   STATION_COORD : 지하철역 좌표 CSV (컬럼: 역명, 위도, 경도)
#   DONG_BOUNDARY : 서울 행정동 경계 GeoJSON (컬럼: adm_nm = '서울특별시 OO구 OO동')
# ==============================================================================

#%% [0] Imports & Config
import pandas as pd
import geopandas as gpd

STATION_COORD = './data/raw/서울교통공사_1_8호선_역사_좌표_위경도__정보_20250814.csv'
DONG_BOUNDARY = './data/raw/서울_행정동_경계_2017.geojson'
OUT_PATH      = './data/station_catchment_weights.csv'

RADIUS_M   = 250          # 역세권 반경(m). 250≈도보 3~4분. 클수록 인접역이 비슷해짐.
COORD_ENC  = 'cp949'      # 좌표 CSV 인코딩
COORD_NAME, COORD_LAT, COORD_LON = '역명', '위도', '경도'
DONG_COL   = 'adm_nm'     # 경계 GeoJSON 행정동명 컬럼
METRIC_CRS = 5186         # 버퍼 연산용 미터 좌표계 (한국 중부원점 TM)


#%% [1] Load & reproject
st = pd.read_csv(STATION_COORD, encoding=COORD_ENC)[[COORD_NAME, COORD_LAT, COORD_LON]] \
       .drop_duplicates(COORD_NAME)
gst = gpd.GeoDataFrame(
    st, geometry=gpd.points_from_xy(st[COORD_LON], st[COORD_LAT]), crs='EPSG:4326'
).to_crs(METRIC_CRS)

dong = gpd.read_file(DONG_BOUNDARY).to_crs(METRIC_CRS)
dong['동'] = dong[DONG_COL].str.split().str[-1]      # '서울특별시 종로구 사직동' → '사직동'
dong = dong[['동', 'geometry']]
print(f'[load] stations={len(gst)} | dongs={len(dong)} | radius={RADIUS_M}m')


#%% [2] Buffer × dong intersection → area-weighted catchment
rows = []
for r in gst.itertuples():
    buf = r.geometry.buffer(RADIUS_M)
    hit = dong[dong.intersects(buf)].copy()
    if hit.empty:
        continue
    hit['a'] = hit.geometry.intersection(buf).area
    hit = hit[hit['a'] > 0]
    total = hit['a'].sum()
    for h in hit.itertuples():
        rows.append({'역명': getattr(r, COORD_NAME), '동': h.동, 'weight': round(h.a / total, 5)})

W = pd.DataFrame(rows)
print(f'[catchment] {W["역명"].nunique()} stations × {len(W)} (station,dong) rows '
      f'| avg {len(W)/W["역명"].nunique():.2f} dongs/station')


#%% [3] (선택) 검증 — 이미 신뢰하는 매핑이 있으면 복원율 확인
# 2호선처럼 정답 매핑을 알면 catchment 가 그 동을 포함하는지로 sanity check.
KNOWN = {  # 예시: 2호선 일부 (역명: 정답 행정동). 전체를 넣으면 더 정확.
    '시청': '소공동', '강남': '역삼1동', '신림': '신림동', '홍대입구': '서교동',
}
if KNOWN:
    inset = {s: set(W[W['역명'] == s]['동']) for s in KNOWN}
    hit = sum(d in inset.get(s, set()) for s, d in KNOWN.items())
    print(f'[check] 알려진 {len(KNOWN)}개 중 catchment 포함 {hit} '
          f'({hit/len(KNOWN):.0%}) — 낮으면 좌표/경계/CRS/동표기 점검')


#%% [4] Save
W.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print(f'[save] → {OUT_PATH}')
print(W[W['역명'] == '강남'].to_string(index=False))
