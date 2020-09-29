#!/bin/zsh
for i in {0..399}; # or something like $(seq 10 10 90) instead of {10,20} if you have a lot of different prefixes
do
	mkdir -p "$((($i / 10)+1))" &&
		mv "$i".png "$((($i / 10)+1))"/"$(( $i + 1))".png
done;
