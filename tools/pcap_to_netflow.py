import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import struct
import socket
from collections import defaultdict
import polars as pl

# 41 NetFlow features required by the GNN model
CATEGORICAL_COLS = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"
]
NUMERICAL_COLS = [
    "L4_SRC_PORT", "L4_DST_PORT", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_QUERY_ID",
    "DNS_TTL_ANSWER"
]
ALL_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"] + CATEGORICAL_COLS + NUMERICAL_COLS


def parse_pcap_file(pcap_path: str) -> list[dict]:
    """Pure Python standard PCAP parser: reads packets and aggregates into NetFlow records."""
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"Không tìm thấy file PCAP tại: {pcap_path}")

    flows = defaultdict(lambda: {
        "start_ts": None, "end_ts": None,
        "pkts": 0, "bytes": 0,
        "ttls": [], "pkt_lens": [],
        "tcp_flags": 0, "tcp_win_max": 0,
        "icmp_type": 0,
    })

    with open(pcap_path, "rb") as f:
        ghdr = f.read(24)
        if len(ghdr) < 24:
            raise ValueError("File PCAP quá ngắn hoặc không đúng định dạng.")

        magic = struct.unpack("!I", ghdr[:4])[0]
        if magic in (0xA1B2C3D4, 0xA1B23C4D):
            endian = ">"
        elif magic in (0xD4C3B2A1, 0x4D3CB2A1):
            endian = "<"
        else:
            # Check pcapng magic 0x0A0D0D0A
            if ghdr[:4] == b"\x0a\x0d\x0d\x0a":
                f.close()
                # Try auto-converting using Wireshark's editcap or tshark
                tshark_paths = [
                    r"C:\Program Files\Wireshark\editcap.exe",
                    r"C:\Program Files\Wireshark\tshark.exe",
                    "editcap",
                    "tshark",
                ]
                converted_pcap = pcap_path + ".converted.pcap"
                for tool in tshark_paths:
                    if os.path.exists(tool) or tool in ("editcap", "tshark"):
                        try:
                            import subprocess
                            if "editcap" in tool.lower():
                                cmd = [tool, "-F", "pcap", pcap_path, converted_pcap]
                            else:
                                cmd = [tool, "-r", pcap_path, "-w", converted_pcap, "-F", "pcap"]
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if os.path.exists(converted_pcap):
                                res = parse_pcap_file(converted_pcap)
                                try:
                                    os.remove(converted_pcap)
                                except Exception:
                                    pass
                                return res
                        except Exception:
                            continue
                raise ValueError("File là định dạng PCAPNG. Vui lòng mở bằng Wireshark và chọn 'Save As' -> chọn loại 'Wireshark/tcpdump/... - pcap (*.pcap)'.")
            endian = "<"

        network_type = struct.unpack(endian + "I", ghdr[20:24])[0]

        while True:
            phdr = f.read(16)
            if len(phdr) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", phdr)
            ts = ts_sec + (ts_usec / 1e6)
            data = f.read(incl_len)

            # Determine link layer offset
            offset = 14 if network_type == 1 else (16 if network_type == 113 else 0)
            if len(data) < offset + 20:
                continue

            # Check if IPv4
            eth_type = struct.unpack("!H", data[12:14])[0] if offset == 14 else 0x0800
            if eth_type != 0x0800:
                continue

            ip_hdr = data[offset:offset + 20]
            ver_ihl, tos, total_len, ident, flags_frag, ttl, proto, checksum, src_raw, dst_raw = struct.unpack("!BBHHHBBH4s4s", ip_hdr)
            ihl = (ver_ihl & 0x0F) * 4
            src_ip = socket.inet_ntoa(src_raw)
            dst_ip = socket.inet_ntoa(dst_raw)

            l4_data = data[offset + ihl:]
            src_port = 0
            dst_port = 0
            tcp_flags = 0
            win_size = 0
            icmp_type = 0

            if proto == 6 and len(l4_data) >= 14:  # TCP
                src_port, dst_port, seq, ack, data_offset_flags, win_size = struct.unpack("!HHIIHH", l4_data[:16])
                tcp_flags = data_offset_flags & 0x00FF
            elif proto == 17 and len(l4_data) >= 8:  # UDP
                src_port, dst_port, udp_len, udp_chk = struct.unpack("!HHHH", l4_data[:8])
            elif proto == 1 and len(l4_data) >= 2:  # ICMP
                icmp_type = l4_data[0]

            key = (src_ip, dst_ip, src_port, dst_port, proto)
            rec = flows[key]
            if rec["start_ts"] is None:
                rec["start_ts"] = ts
            rec["end_ts"] = ts
            rec["pkts"] += 1
            rec["bytes"] += total_len
            rec["ttls"].append(ttl)
            rec["pkt_lens"].append(total_len)
            rec["tcp_flags"] |= tcp_flags
            rec["tcp_win_max"] = max(rec["tcp_win_max"], win_size)
            if icmp_type:
                rec["icmp_type"] = icmp_type

    # Format into 41-feature records
    records = []
    for (src_ip, dst_ip, src_port, dst_port, proto), stat in flows.items():
        dur_ms = max(0.0, (stat["end_ts"] - stat["start_ts"]) * 1000.0)
        ttls = stat["ttls"] or [64]
        pkt_lens = stat["pkt_lens"] or [60]

        rec_dict = {
            "IPV4_SRC_ADDR": src_ip,
            "IPV4_DST_ADDR": dst_ip,
            "PROTOCOL": int(proto),
            "L7_PROTO": 0,
            "TCP_FLAGS": int(stat["tcp_flags"]),
            "CLIENT_TCP_FLAGS": int(stat["tcp_flags"]),
            "SERVER_TCP_FLAGS": 0,
            "ICMP_TYPE": int(stat["icmp_type"]),
            "ICMP_IPV4_TYPE": int(stat["icmp_type"]),
            "DNS_QUERY_TYPE": 1 if dst_port == 53 or src_port == 53 else 0,
            "FTP_COMMAND_RET_CODE": 0,
            "L4_SRC_PORT": float(src_port),
            "L4_DST_PORT": float(dst_port),
            "IN_BYTES": float(stat["bytes"]),
            "IN_PKTS": float(stat["pkts"]),
            "OUT_BYTES": 0.0,
            "OUT_PKTS": 0.0,
            "FLOW_DURATION_MILLISECONDS": float(dur_ms),
            "DURATION_IN": float(dur_ms),
            "DURATION_OUT": 0.0,
            "MIN_TTL": float(min(ttls)),
            "MAX_TTL": float(max(ttls)),
            "LONGEST_FLOW_PKT": float(max(pkt_lens)),
            "SHORTEST_FLOW_PKT": float(min(pkt_lens)),
            "MIN_IP_PKT_LEN": float(min(pkt_lens)),
            "MAX_IP_PKT_LEN": float(max(pkt_lens)),
            "SRC_TO_DST_SECOND_BYTES": float(stat["bytes"] / max(dur_ms / 1000.0, 1.0)),
            "DST_TO_SRC_SECOND_BYTES": 0.0,
            "RETRANSMITTED_IN_BYTES": 0.0,
            "RETRANSMITTED_IN_PKTS": 0.0,
            "RETRANSMITTED_OUT_BYTES": 0.0,
            "RETRANSMITTED_OUT_PKTS": 0.0,
            "SRC_TO_DST_AVG_THROUGHPUT": float(stat["bytes"] * 8.0 / max(dur_ms / 1000.0, 0.001)),
            "DST_TO_SRC_AVG_THROUGHPUT": 0.0,
            "NUM_PKTS_UP_TO_128_BYTES": float(sum(1 for l in pkt_lens if l <= 128)),
            "NUM_PKTS_128_TO_256_BYTES": float(sum(1 for l in pkt_lens if 128 < l <= 256)),
            "NUM_PKTS_256_TO_512_BYTES": float(sum(1 for l in pkt_lens if 256 < l <= 512)),
            "NUM_PKTS_512_TO_1024_BYTES": float(sum(1 for l in pkt_lens if 512 < l <= 1024)),
            "NUM_PKTS_1024_TO_1514_BYTES": float(sum(1 for l in pkt_lens if 1024 < l <= 1514)),
            "TCP_WIN_MAX_IN": float(stat["tcp_win_max"]),
            "TCP_WIN_MAX_OUT": 0.0,
            "DNS_QUERY_ID": 0.0,
            "DNS_TTL_ANSWER": 0.0,
        }
        records.append(rec_dict)

    return records


def main():
    if len(sys.argv) < 2:
        print("Sử dụng: python tools/pcap_to_netflow.py <duong_dan_file.pcap> [file_xuat.csv]")
        print("Ví dụ:   python tools/pcap_to_netflow.py capture_that.pcap real_traffic.csv")
        return

    pcap_in = sys.argv[1]
    csv_out = sys.argv[2] if len(sys.argv) > 2 else "real_world_traffic.csv"

    print(f"[*] Đang đọc và phân tích luồng từ file PCAP thực tế: {pcap_in}...")
    try:
        records = parse_pcap_file(pcap_in)
    except Exception as e:
        print(f"[!] Lỗi: {e}")
        return

    if not records:
        print("[!] Không tìm thấy gói tin IPv4 nào trong file PCAP.")
        return

    df = pl.DataFrame(records)
    df.write_csv(csv_out)
    print(f"[+] Thành công! Đã trích xuất {len(df)} luồng NetFlow thực tế sang: {csv_out}")
    print(f"[+] Bạn có thể tải trực tiếp file '{csv_out}' lên Web Console (http://127.0.0.1:8000/)!")


if __name__ == "__main__":
    main()
