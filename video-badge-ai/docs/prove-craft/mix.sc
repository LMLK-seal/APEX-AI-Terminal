session.new {"name":"prova","sample_rate":48000,"bit_depth":24}
track.new {"name":"Voce","format":"mono"}
track.new {"name":"Musica","format":"mono"}
file.import_audio {"path":"C:/Users/franc/progetto-p-Iva/video-badge-ai/docs/prove-craft/voce_test.wav","track":"Voce","at":"0:01.000","new_tracks":false}
file.import_audio {"path":"C:/Users/franc/progetto-p-Iva/video-badge-ai/docs/prove-craft/musica_test.wav","track":"Musica","at":0,"new_tracks":false}
mix.volume {"tracks":["Musica"],"db":-12}
