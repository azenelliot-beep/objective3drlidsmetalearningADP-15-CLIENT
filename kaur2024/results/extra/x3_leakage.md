# X3 [EXTRA] Leakage checks

File `TON_IoT_Train_Test_Network.csv`, split from partition seed 0.

## Cardinality of every string column (whole file, report only)

| column | n_unique | top_values | G2 action |
|---|---|---|---|
| src_ip | 11536 | 192.168.1.190, 192.168.1.152, 192.168.1.30, 192.168.1.79, 192.168.1.195 | drop (G2 list) |
| dst_ip | 5268 | 192.168.1.190, 192.168.1.152, 192.168.1.255, 192.168.1.195, 192.168.1.193 | drop (G2 list) |
| proto | 3 | tcp, udp, icmp | one-hot |
| service | 10 | -, dns, http, ssl, ftp | one-hot |
| conn_state | 13 | SF, S0, OTH, REJ, SHR | one-hot |
| dns_query | 14149 | -, a2z3kk2ebqzso7.iot.ap-southeast-2.amazonaws.com, arpa.whois-servers.net, _sleep-proxy._udp.local, testphp.vulnweb.com | drop (>50 unique) |
| dns_AA | 3 | -, F, T | one-hot |
| dns_RD | 3 | -, F, T | one-hot |
| dns_RA | 3 | -, F, T | one-hot |
| dns_rejected | 3 | -, F, T | one-hot |
| ssl_version | 4 | -, TLSv12, TLSv13, TLSv10 | one-hot |
| ssl_cipher | 6 | -, TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, TLS_AES_128_GCM_SHA256, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384, TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA | one-hot |
| ssl_resumed | 3 | -, F, T | one-hot |
| ssl_established | 3 | -, F, T | one-hot |
| ssl_subject | 6 | -, CN=*.wns.windows.com, CN=Mr Shepherd;OU=Security Shepherd Project;O=OWASP;L=Dublin;ST=Ireland;C=IE, CN=settings-win.data.microsoft.com;OU=WSE;O=Microsoft;L=Redmond;ST=WA;C=US, CN=sls.update.microsoft.com;OU=DSP;O=Microsoft;L=Redmond;ST=WA;C=US | one-hot |
| ssl_issuer | 5 | -, CN=Microsoft IT TLS CA 5;OU=Microsoft IT;O=Microsoft Corporation;L=Redmond;ST=Washington;C=US, CN=Microsoft Secure Server CA 2011;O=Microsoft Corporation;L=Redmond;ST=Washington;C=US, CN=Mr Shepherd;OU=Security Shepherd Project;O=OWASP;L=Dublin;ST=Ireland;C=IE, CN=Microsoft Update Secure Server CA 2.1;O=Microsoft Corporation;L=Redmond;ST=Washington;C=US | one-hot |
| http_trans_depth | 11 | -, 1, 2, 10, 3 | one-hot |
| http_method | 4 | -, GET, POST, HEAD | one-hot |
| http_uri | 74 | -, /no_nonce_string/, /dvwa/login.php-r, /no_nonce_string/ContentDirectory/scpd.xml, /filestreamingservice/files/846ab63f-c24f-4c71-bc98-6d8fa3f5e5d7?P1=1554310453&P2=402&P3=2&P4=l0mwXDvb5/y2rqu/g6TNhETQLS++OvWqWkMXYMSnMuqiHgfVXg9rPHQAr/I7U+Oo5I++LRwjHuFY5S5SbIij1g== | drop (>50 unique) |
| http_version | 2 | -, 1.1 | one-hot |
| http_user_agent | 36 | -, Microsoft-Delivery-Optimization/10.0, Microsoft-Windows/10.0 UPnP/1.0, Ruby, DAFUPnP | one-hot |
| http_orig_mime_types | 3 | -, application/xml, application/soap+xml | one-hot |
| http_resp_mime_types | 10 | -, application/xml, text/html, image/png, application/vnd.ms-cab-compressed | one-hot |
| weird_name | 12 | -, DNS_RR_unknown_type, active_connection_reuse, data_before_established, bad_UDP_checksum | one-hot |
| weird_addl | 4 | -, 46, 48, 43 | one-hot |
| weird_notice | 2 | -, F | one-hot |
| type | 10 | normal, scanning, dos, injection, ddos | target |

## Duplicate rows shared between train and test

### Key: all columns except `type`/`label`
- test rows whose key also occurs in train: 2,708 / 89,518 (3.03%)
- of those, with a different `type` in train: 0
- duplicates within train: 8,160; within test: 948
- shared test rows by class: normal 2,708

### Key: model features (after G2 drops, before encoding)
- test rows whose key also occurs in train: 54,064 / 89,518 (60.39%)
- of those, with a different `type` in train: 8,909
- duplicates within train: 220,061; within test: 49,743
- shared test rows by class: normal 33,948, ransomware 3,963, backdoor 3,710, xss 3,484, scanning 3,117, dos 2,883, ddos 1,789, password 1,109, injection 57, mitm 4
