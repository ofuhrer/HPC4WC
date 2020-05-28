docker stop xue_filter 1>/dev/null 2>/dev/null
docker rm xue_filter 1>/dev/null 2>/dev/null
docker run -i -t --rm --mount type=bind,source=`pwd`/work,target=/work --name=xue_filter xue_filter
