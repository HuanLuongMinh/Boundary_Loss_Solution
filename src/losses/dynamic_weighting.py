"""src/losses/dynamic_weighting.py — lịch trình lambda1(t)/lambda2(t) cho
Run 4 "Dynamic Boundary-Aware Loss" (docs/spec-run4-dynamic-weighting.md).

File MỚI, ĐỘC LẬP HOÀN TOÀN với src/losses/total_loss.py (Run 3) và
src/losses/total_loss_weighted.py (Run 3b) — không import, không sửa 2 file
đó. src/losses/total_loss_dynamic.py (Run 4) import các hàm ở đây.

Hai lịch trình:

  lambda_schedule            — lịch trình gốc mô tả ở spec mục 0.1
                                (lambda1 giảm tuyến tính 1.0->0.3, lambda2
                                tăng tuyến tính 0.0->1.0, suốt toàn bộ run).
                                Bị bác bỏ bởi dữ liệu Run 3b (spec mục 0.1) —
                                giữ lại CHỈ để đăng ký trong SCHEDULES cho đầy
                                đủ/tái lập được, Run 4 KHÔNG dùng lịch trình
                                này (config Run 4 chọn 'ramp_hold').

  lambda_schedule_ramp_hold  — lịch trình CHỌN cho Run 4 (spec mục 1):
                                lambda1 phẳng 1.0; lambda2 giữ 0 trong
                                warmup_frac đầu, ramp tuyến tính lên
                                lambda2_end tới ramp_end_frac, rồi giữ
                                nguyên (hold) tới hết run.
"""

SCHEDULES = {}


def lambda_schedule(cur_iter, max_iters,
                     lambda1_start=1.0, lambda1_end=0.3,
                     lambda2_start=0.0, lambda2_end=1.0):
    """Lịch trình gốc (spec mục 0.1) — lambda1 giảm tuyến tính, lambda2 tăng
    tuyến tính, suốt toàn bộ [0, max_iters]. KHÔNG dùng cho Run 4 (đã bác bỏ
    bằng dữ liệu Run 3b: giảm lambda1 làm yếu BCE vô căn cứ, và ramp lambda2
    tuyến tính suốt run khiến phần lớn thời gian huấn luyện rơi vào vùng liều
    affinity thấp đã đo là có hại). Giữ lại nguyên vẹn để mọi thứ tái lập
    được và để SCHEDULES['linear'] có nghĩa."""
    p = cur_iter / max(max_iters, 1)
    p = min(max(p, 0.0), 1.0)
    lam1 = lambda1_start + (lambda1_end - lambda1_start) * p
    lam2 = lambda2_start + (lambda2_end - lambda2_start) * p
    return lam1, lam2


def lambda_schedule_ramp_hold(cur_iter, max_iters,
                               lambda1_const=1.0,
                               lambda2_end=1.0,
                               warmup_frac=0.10,
                               ramp_end_frac=0.30):
    """lambda1 phẳng; lambda2 giữ 0 trong warmup, ramp tuyến tính tới
    lambda2_end, rồi giữ nguyên tới hết. Vượt nhanh qua vùng liều thấp đã đo
    là có hại (spec mục 1, công thức copy nguyên văn)."""
    p = cur_iter / max(max_iters, 1)
    lam1 = lambda1_const
    if p < warmup_frac:
        lam2 = 0.0
    elif p < ramp_end_frac:
        lam2 = lambda2_end * (p - warmup_frac) / (ramp_end_frac - warmup_frac)
    else:
        lam2 = lambda2_end
    return lam1, lam2


SCHEDULES.update({
    'linear':    lambda_schedule,            # bản gốc, giữ nguyên — KHÔNG dùng ở Run 4
    'ramp_hold': lambda_schedule_ramp_hold,  # MỚI — dùng cho Run 4
})


if __name__ == "__main__":
    # Self-test — chạy `python -m src.losses.dynamic_weighting` từ repo root.
    # Kiểm đúng bảng đóng ở docs/spec-run4-dynamic-weighting.md mục 5.1.
    MAX_ITERS = 40000
    table = [
        (0,     1.0, 0.0),
        (3999,  1.0, 0.0),
        (4000,  1.0, 0.0),
        (8000,  1.0, 0.5),
        (12000, 1.0, 1.0),
        (40000, 1.0, 1.0),
    ]
    for cur_iter, exp_lam1, exp_lam2 in table:
        lam1, lam2 = lambda_schedule_ramp_hold(cur_iter, MAX_ITERS)
        assert abs(lam1 - exp_lam1) < 1e-6, f"cur_iter={cur_iter}: lam1={lam1} != {exp_lam1}"
        assert abs(lam2 - exp_lam2) < 1e-6, f"cur_iter={cur_iter}: lam2={lam2} != {exp_lam2}"
        print(f"cur_iter={cur_iter:>6}  lambda1={lam1:.4f}  lambda2={lam2:.4f}  OK")
    print("PASS — lambda_schedule_ramp_hold khớp đúng bảng đóng mục 5.1.")

    # SCHEDULES registry đúng 2 khoá, trỏ đúng hàm.
    assert set(SCHEDULES.keys()) == {'linear', 'ramp_hold'}
    assert SCHEDULES['linear'] is lambda_schedule
    assert SCHEDULES['ramp_hold'] is lambda_schedule_ramp_hold

    # lambda_schedule (bản gốc) — sanity nhanh: 2 đầu mút + trung điểm tuyến tính.
    lam1_0, lam2_0 = lambda_schedule(0, MAX_ITERS)
    assert abs(lam1_0 - 1.0) < 1e-9 and abs(lam2_0 - 0.0) < 1e-9
    lam1_end, lam2_end = lambda_schedule(MAX_ITERS, MAX_ITERS)
    assert abs(lam1_end - 0.3) < 1e-9 and abs(lam2_end - 1.0) < 1e-9
    lam1_mid, lam2_mid = lambda_schedule(MAX_ITERS // 2, MAX_ITERS)
    assert abs(lam1_mid - 0.65) < 1e-9 and abs(lam2_mid - 0.5) < 1e-9
    print("PASS — lambda_schedule (ban goc, KHONG dung o Run 4) dung cong thuc tuyen tinh.")

    print("PASS — SCHEDULES registry dung 2 khoa 'linear'/'ramp_hold'.")
