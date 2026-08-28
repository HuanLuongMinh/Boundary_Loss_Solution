"""
Xác định checkpoint để resume training (dùng cho scripts/resume_baseline.sh).

Có 2 cách lấy checkpoint, tùy môi trường chạy:

1. CHẾ ĐỘ TERMINAL (mặc định — dùng được khi gọi qua subprocess/bash, vd
   scripts/resume_baseline.sh): hỏi người dùng nhập đường dẫn file đã upload (qua
   panel "Data" của Kaggle hoặc File Browser của notebook), rồi xác nhận file
   tồn tại. Mọi hướng dẫn in ra STDERR; dòng cuối in ra STDOUT là đường dẫn
   checkpoint hợp lệ duy nhất — để bash lấy bằng $(...) mà không lẫn hướng dẫn.

       python Tools/get_resume_checkpoint.py --default-path /kaggle/working/.../latest_checkpoint.pth
       python Tools/get_resume_checkpoint.py --default-path ... --path /kaggle/working/foo.pth   # bỏ qua hỏi

2. CHẾ ĐỘ BROWSE/UPLOAD TRÊN TRÌNH DUYỆT (chỉ dùng được khi chạy TRỰC TIẾP
   trong 1 cell của Kaggle/Jupyter notebook — qua `%run` hoặc import, KHÔNG
   chạy được qua `!python ...` hay qua bash/subprocess, vì widget cần kết nối
   Comm trực tiếp tới kernel đang hiển thị trên trình duyệt của bạn):

       %run "Tools/get_resume_checkpoint.py" --browser --default-path /kaggle/working/.../latest_checkpoint.pth

   Lệnh này hiện ra 1 nút "Upload" — bấm vào sẽ mở hộp thoại browse file của
   trình duyệt, chọn file checkpoint trên máy bạn, file sẽ được tự động lưu
   vào đúng đường dẫn --default-path. Sau khi thấy dòng "✔ Đã lưu checkpoint",
   chạy `bash scripts/resume_baseline.sh` như bình thường (sẽ tự thấy file).
"""

import argparse
import os
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def browse_upload_checkpoint(save_path: str):
    """Hiển thị widget upload (mở hộp thoại browse trên trình duyệt) để chọn
    file checkpoint, rồi tự lưu vào `save_path` khi người dùng upload xong.

    CHỈ hoạt động khi gọi trực tiếp trong 1 cell Jupyter/Kaggle notebook (hoặc
    qua %run) — không hoạt động khi script chạy qua subprocess/bash, vì widget
    cần kết nối Comm trực tiếp tới kernel đang hiển thị trên trình duyệt.
    """
    if not in_jupyter():
        raise RuntimeError(
            'browse_upload_checkpoint() chỉ hoạt động khi chạy trong 1 cell '
            'Jupyter/Kaggle notebook (hoặc qua %run) — không hoạt động qua '
            'subprocess/bash (vd scripts/resume_baseline.sh).'
        )

    import ipywidgets as widgets
    from IPython.display import display

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    uploader = widgets.FileUpload(accept='.pth', multiple=False,
                                  description='Upload checkpoint')
    label = widgets.Label(value=f'Chọn file checkpoint để upload (sẽ lưu vào: {save_path})')

    def _on_upload(change):
        new_files = change['new']
        if not new_files:
            return
        # ipywidgets >= 8: dict {filename: {...}};  < 8: tuple of dicts
        fileinfo = next(iter(new_files.values())) if isinstance(new_files, dict) else new_files[0]
        content = fileinfo['content']
        with open(save_path, 'wb') as f:
            f.write(bytes(content))
        print(f'✔ Đã lưu checkpoint vào: {save_path}')

    uploader.observe(_on_upload, names='value')
    display(label, uploader)
    return uploader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default-path', required=True,
                        help='Đường dẫn checkpoint gợi ý mặc định trong WORK_DIR')
    parser.add_argument('--path', default=None,
                        help='Bỏ qua hỏi, dùng luôn đường dẫn này (phải tồn tại)')
    parser.add_argument('--browser', action='store_true',
                        help='Hiện widget upload trên trình duyệt (chỉ dùng trong '
                             'notebook cell qua %%run, không dùng qua bash/subprocess)')
    args = parser.parse_args()

    if args.browser:
        browse_upload_checkpoint(args.default_path)
        eprint('Widget upload đã hiển thị phía trên. Sau khi upload xong, chạy lại')
        eprint('scripts/resume_baseline.sh (script sẽ tự nhận ra file vừa lưu).')
        return

    if args.path:
        if not os.path.exists(args.path):
            eprint(f'Lỗi: checkpoint không tồn tại: {args.path}')
            sys.exit(1)
        print(os.path.abspath(args.path))
        return

    if os.path.exists(args.default_path):
        eprint(f'✔ Tìm thấy checkpoint có sẵn: {args.default_path}')
        print(os.path.abspath(args.default_path))
        return

    eprint('============================================================')
    eprint(' Cần checkpoint để resume training.')
    eprint('============================================================')
    eprint(' Vui lòng upload file checkpoint (vd: latest_checkpoint.pth)')
    eprint(f' lên đúng đường dẫn:')
    eprint(f'   {args.default_path}')
    eprint(' Trên Kaggle: mở panel "Data" bên phải > Upload, hoặc dùng')
    eprint(' File Browser của notebook để kéo-thả file vào /kaggle/working/.')
    eprint(' (Hoặc dùng chế độ browse trên trình duyệt — xem docstring đầu file')
    eprint('  để chạy `%run Tools/get_resume_checkpoint.py --browser ...` trong 1 cell.)')
    eprint('============================================================')

    while True:
        eprint('')
        eprint(f'Nhập đường dẫn checkpoint đã upload (Enter để dùng mặc định: {args.default_path}):')
        try:
            entered = input('> ').strip()
        except EOFError:
            eprint('Lỗi: không có input tương tác (EOF). Hãy chạy script ở terminal có stdin,')
            eprint('hoặc dùng --path <checkpoint> để bỏ qua bước hỏi.')
            sys.exit(1)

        path = entered if entered else args.default_path

        if os.path.exists(path):
            eprint(f'✔ Tìm thấy checkpoint: {path}')
            print(os.path.abspath(path))
            return

        eprint(f'✘ Không tìm thấy file: {path}')
        eprint('  Hãy upload file rồi thử lại (hoặc Ctrl-C để dừng).')


if __name__ == '__main__':
    main()
