#!/usr/bin/env bash
set -u

for url in \
  https://pypi.tuna.tsinghua.edu.cn/simple/torch/ \
  https://mirrors.aliyun.com/pypi/simple/torch/ \
  https://pypi.org/simple/torch/
do
  printf '%s\n' "$url"
  curl -IL --connect-timeout 10 --max-time 30 -o /dev/null -sS \
    -w 'http=%{http_code} remote=%{remote_ip} time=%{time_total}\n' "$url" \
    || true
done

for host in \
  pypi.tuna.tsinghua.edu.cn \
  mirrors.aliyun.com \
  pypi.org \
  files.pythonhosted.org
do
  printf '%s=' "$host"
  getent ahostsv4 "$host" | head -1 || true
done
