import sys
import urllib.request
import struct
import zlib
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config_loader import get_storage_dir

def download_wiki_partitions(targets=['wiki-pages/wiki-002.jsonl', 'wiki-pages/wiki-003.jsonl']):
    url = 'https://fever.ai/download/fever/wiki-pages.zip'
    target_dir = get_storage_dir('data/raw', for_writing=True) / 'wiki-pages'
    target_dir.mkdir(parents=True, exist_ok=True)
    
    cd_offset = 1713465715
    cd_size = 19737
    req = urllib.request.Request(url, headers={'Range': f'bytes={cd_offset}-{cd_offset+cd_size-1}'})
    with urllib.request.urlopen(req) as resp:
        cd_data = resp.read()

    pos = 0
    entries = {}
    while pos < len(cd_data):
        if cd_data[pos:pos+4] != b'\x50\x4b\x01\x02':
            break
        fname_len = struct.unpack('<H', cd_data[pos+28:pos+30])[0]
        extra_len = struct.unpack('<H', cd_data[pos+30:pos+32])[0]
        comment_len = struct.unpack('<H', cd_data[pos+32:pos+34])[0]
        comp_size = struct.unpack('<I', cd_data[pos+20:pos+24])[0]
        uncomp_size = struct.unpack('<I', cd_data[pos+24:pos+28])[0]
        local_offset = struct.unpack('<I', cd_data[pos+42:pos+46])[0]
        fname = cd_data[pos+46:pos+46+fname_len].decode('utf-8', errors='ignore')
        if fname in targets:
            entries[fname] = (local_offset, comp_size, uncomp_size)
        pos += 46 + fname_len + extra_len + comment_len

    for fname, (loc_off, comp_sz, uncomp_sz) in entries.items():
        out_name = Path(fname).name
        dest = target_dir / out_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f'{out_name} already exists ({dest.stat().st_size / (1024*1024):.1f} MB). Skipping.')
            continue
        print(f'Extracting {out_name} ({comp_sz/(1024*1024):.1f} MB compressed)...')
        t0 = time.time()
        # Read local header to find data offset
        req = urllib.request.Request(url, headers={'Range': f'bytes={loc_off}-{loc_off+130}'})
        with urllib.request.urlopen(req) as resp:
            hdr = resp.read()
        f_len = struct.unpack('<H', hdr[26:28])[0]
        e_len = struct.unpack('<H', hdr[28:30])[0]
        d_start = loc_off + 30 + f_len + e_len
        d_end = d_start + comp_sz - 1
        
        req = urllib.request.Request(url, headers={'Range': f'bytes={d_start}-{d_end}'})
        with urllib.request.urlopen(req) as resp:
            comp_data = resp.read()
            
        decomp = zlib.decompress(comp_data, -15)
        with open(dest, 'wb') as f:
            f.write(decomp)
        elapsed = time.time() - t0
        print(f'Saved {dest.name} ({len(decomp)/(1024*1024):.1f} MB uncompressed) in {elapsed:.1f}s.')

if __name__ == '__main__':
    download_wiki_partitions()
