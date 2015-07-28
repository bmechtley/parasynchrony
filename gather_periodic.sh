#!/bin/env bash

if [ -z $3 ]; then
	echo "usage: $0 interval percent cmd ..."
fi

while true; do
	a="$(myquota | grep scratch | tr -s '[[:space:]]' '\n' | sed '5q;d' | sed 's/%//')"
	if ! echo "$a $2 -p" | dc | grep > /dev/null ^-; then
		${@:3}
	fi
	sleep $1
done
