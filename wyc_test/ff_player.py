from ffpyplayer.player import MediaPlayer
import time
filename = 'C:\\Users\\16200\\Desktop\\project_dataset\\soccer.mp4'
player = MediaPlayer(filename)
val = ''
while val != 'eof':
    frame, val = player.get_frame()
    if val != 'eof' and frame is not None:
        img, t = frame
        # display img