# Phase 1 data report: `TON_IoT_Train_Test_Network.csv`

- sha256: `65d5465df1809b984fd10e4703bc9c012d1aefd11803773856364216f0a3520d`
- shape: 461,043 rows x 45 columns

## Columns, dtypes, missing values
`n_dash` counts the '-' placeholder TON_IoT uses for 'not applicable'.

| column | dtype | n_nan | n_dash |
|---|---|---|---|
| ts | int64 | 0 | 0 |
| src_ip | object | 0 | 0 |
| src_port | int64 | 0 | 0 |
| dst_ip | object | 0 | 0 |
| dst_port | int64 | 0 | 0 |
| proto | object | 0 | 0 |
| service | object | 0 | 280216 |
| duration | float64 | 0 | 0 |
| src_bytes | int64 | 0 | 0 |
| dst_bytes | int64 | 0 | 0 |
| conn_state | object | 0 | 0 |
| missed_bytes | int64 | 0 | 0 |
| src_pkts | int64 | 0 | 0 |
| src_ip_bytes | int64 | 0 | 0 |
| dst_pkts | int64 | 0 | 0 |
| dst_ip_bytes | int64 | 0 | 0 |
| dns_query | object | 0 | 366019 |
| dns_qclass | int64 | 0 | 0 |
| dns_qtype | int64 | 0 | 0 |
| dns_rcode | int64 | 0 | 0 |
| dns_AA | object | 0 | 365158 |
| dns_RD | object | 0 | 365158 |
| dns_RA | object | 0 | 365158 |
| dns_rejected | object | 0 | 365158 |
| ssl_version | object | 0 | 460737 |
| ssl_cipher | object | 0 | 460737 |
| ssl_resumed | object | 0 | 460352 |
| ssl_established | object | 0 | 460352 |
| ssl_subject | object | 0 | 461034 |
| ssl_issuer | object | 0 | 461034 |
| http_trans_depth | object | 0 | 460796 |
| http_method | object | 0 | 460809 |
| http_uri | object | 0 | 460809 |
| http_version | object | 0 | 460801 |
| http_request_body_len | int64 | 0 | 0 |
| http_response_body_len | int64 | 0 | 0 |
| http_status_code | int64 | 0 | 0 |
| http_user_agent | object | 0 | 460809 |
| http_orig_mime_types | object | 0 | 461029 |
| http_resp_mime_types | object | 0 | 460883 |
| weird_name | object | 0 | 459749 |
| weird_addl | object | 0 | 460290 |
| weird_notice | object | 0 | 459749 |
| label | int64 | 0 | 0 |
| type | object | 0 | 0 |

## Class counts vs Table II [PAPER, p.8] and the split [GAP-FILL G1]

| class | in_file | paper_train | ours_train | paper_test | ours_test | unused | short_train | short_test |
|---|---|---|---|---|---|---|---|---|
| Normal | 300000 | 245000 | 245000 | 55000 | 55000 | 0 | 0 | 0 |
| Scanning | 20000 | 20000 | 16027 | 3973 | 3973 | 0 | 3973 | 0 |
| DoS | 20000 | 20000 | 16000 | 4000 | 4000 | 0 | 4000 | 0 |
| DDoS | 20000 | 20000 | 16000 | 4000 | 4000 | 0 | 4000 | 0 |
| Ransomware | 20000 | 16030 | 16030 | 3970 | 3970 | 0 | 0 | 0 |
| Backdoor | 20000 | 20000 | 16000 | 4000 | 4000 | 0 | 4000 | 0 |
| Injection Attack | 20000 | 20000 | 16000 | 4000 | 4000 | 0 | 4000 | 0 |
| XSS | 20000 | 13844 | 13844 | 6116 | 6116 | 40 | 0 | 0 |
| Password Violations | 20000 | 20000 | 16000 | 4000 | 4000 | 0 | 4000 | 0 |
| MITM | 1043 | 593 | 584 | 459 | 459 | 0 | 9 | 0 |

Totals: train 371,485 (paper 395,467), test 89,518 (paper 89,518).
`type` values not in Table II: none

## Client partition: iid

| index | Normal | Scanning | DoS | DDoS | Ransomware | Backdoor | Injection Attack | XSS | Password Violations | MITM | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| client_00 | 16252 | 1056 | 1066 | 1080 | 1099 | 1063 | 1132 | 927 | 1054 | 37 | 24766 |
| client_01 | 16416 | 1077 | 1006 | 1041 | 1052 | 1125 | 1051 | 915 | 1053 | 30 | 24766 |
| client_02 | 16431 | 1083 | 1048 | 1071 | 988 | 1080 | 1083 | 897 | 1044 | 41 | 24766 |
| client_03 | 16322 | 1057 | 1113 | 1078 | 1110 | 1007 | 1061 | 936 | 1040 | 42 | 24766 |
| client_04 | 16314 | 1052 | 1052 | 1043 | 1079 | 1031 | 1050 | 972 | 1117 | 56 | 24766 |
| client_05 | 16244 | 1101 | 1078 | 1109 | 1065 | 1102 | 1074 | 909 | 1050 | 34 | 24766 |
| client_06 | 16282 | 1061 | 1113 | 1094 | 1066 | 1096 | 1066 | 925 | 1032 | 31 | 24766 |
| client_07 | 16368 | 1030 | 1020 | 1068 | 1046 | 1112 | 1052 | 928 | 1100 | 42 | 24766 |
| client_08 | 16302 | 1096 | 1037 | 1087 | 1084 | 1050 | 1083 | 909 | 1083 | 35 | 24766 |
| client_09 | 16342 | 1073 | 1036 | 1044 | 1058 | 1098 | 1043 | 949 | 1089 | 34 | 24766 |
| client_10 | 16319 | 1082 | 1131 | 1049 | 1048 | 1043 | 1056 | 920 | 1065 | 52 | 24765 |
| client_11 | 16413 | 1080 | 1072 | 1024 | 1083 | 1015 | 1049 | 926 | 1068 | 35 | 24765 |
| client_12 | 16323 | 1056 | 1090 | 1076 | 1082 | 1073 | 1074 | 934 | 1023 | 34 | 24765 |
| client_13 | 16238 | 1059 | 1091 | 1051 | 1127 | 1082 | 1075 | 892 | 1102 | 48 | 24765 |
| client_14 | 16434 | 1064 | 1047 | 1085 | 1043 | 1023 | 1051 | 905 | 1080 | 33 | 24765 |

## Client partition: noniid_dir0.5

| index | Normal | Scanning | DoS | DDoS | Ransomware | Backdoor | Injection Attack | XSS | Password Violations | MITM | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| client_00 | 3861 | 761 | 162 | 72 | 1464 | 130 | 1902 | 124 | 1495 | 142 | 10113 |
| client_01 | 36311 | 119 | 44 | 2 | 113 | 598 | 85 | 865 | 89 | 21 | 38247 |
| client_02 | 6376 | 140 | 1149 | 4585 | 976 | 1037 | 110 | 695 | 7515 | 137 | 22720 |
| client_03 | 946 | 2094 | 1724 | 16 | 3221 | 128 | 8260 | 2458 | 632 | 3 | 19482 |
| client_04 | 1531 | 72 | 5473 | 645 | 7102 | 1058 | 809 | 253 | 2389 | 39 | 19371 |
| client_05 | 3049 | 668 | 277 | 3283 | 1137 | 1531 | 1142 | 7 | 78 | 5 | 11177 |
| client_06 | 26123 | 3089 | 450 | 58 | 288 | 522 | 1000 | 245 | 94 | 20 | 31889 |
| client_07 | 1808 | 2443 | 101 | 284 | 144 | 1544 | 187 | 5 | 1 | 0 | 6517 |
| client_08 | 10 | 217 | 2710 | 15 | 70 | 2 | 292 | 902 | 2 | 5 | 4225 |
| client_09 | 41783 | 259 | 1002 | 1826 | 499 | 1502 | 390 | 4417 | 102 | 47 | 51827 |
| client_10 | 2555 | 54 | 1921 | 2342 | 362 | 881 | 22 | 1782 | 0 | 61 | 9980 |
| client_11 | 18330 | 3364 | 760 | 1423 | 292 | 3215 | 408 | 566 | 80 | 34 | 28472 |
| client_12 | 3343 | 68 | 6 | 779 | 229 | 20 | 1010 | 606 | 2719 | 67 | 8847 |
| client_13 | 83180 | 1016 | 126 | 58 | 33 | 2161 | 22 | 846 | 472 | 1 | 87915 |
| client_14 | 15794 | 1663 | 95 | 612 | 100 | 1671 | 361 | 73 | 332 | 2 | 20703 |

Dirichlet draws needed to reach min client size 20: 1
