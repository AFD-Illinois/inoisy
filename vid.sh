# input should be name of folder, without the '/' at end

if [ "$1" != "" ]; then
		ffmpeg -r 24 -f image2 -i $1/%d.png -vcodec libx264 -crf 25 $(dirname $1)/$(basename $1).mp4
else
		echo "Folder path is empty"
fi
