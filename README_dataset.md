This is monologue dataset with self-report questionnaire scores.

######################
### Specifications ###
######################

- Language: Japanese
- Number of participants: 55
- Number of audio/transcript data: 454

- Audio duration:
--- session[1,2,3]: 30 seconds
--- session[4,5,6]: 60 seconds
--- session[7,8,9]: 180 seconds

- Talk theme: We asked to talk about favorite thing if they don't remember recent dream at night.
--- session[1,4,7]: Recent dream at night (see data_label/dream_flag.csv)
--- session[1,4,7]: Favorite thing (see data_label/dream_flag.csv)
--- session[2,5,8]: Negative memory
--- session[3,6,9]: The biggest mistake

- Speech phase:
--- Phase id 1: 0-60 seconds
--- Phase id 2: 60-120 seconds
--- Phase id 3: 120-180 seconds


####################
### Score labels ###
####################

- BigFive: BigFive personalities
- KiSS18: Kikuchi's scale of social skills 18 items
- SPQ: Schizotypal personality Questionnaire
- SRS2: 2nd edition of social responsiveness scale


################
### Contents ###
################

- data_label:
--- dream_flag.csv: flag whether the corresponding topic is "recent dream at night" or "favorite thing".
--- score: CSV format
----- complete: subjects data which has all of score labels.
----- incomplete: subjects data which has at least some of score labels (some people didn't complete to fill some of questionnaires).

- data_audio: 
--- format: Record_[user id]_[recording date]_[session id].mp3

- data_audio_separated180: 
--- format: Record_[user id]_[recording date]_[session id]_[phase id].mp3

- data_transcript:
--- format: Record_[user id]_[recording date]_[session id].mp3

- data_transcript_separated180
--- format: Record_[user id]_[recording date]_[session id]_[phase id].mp3


For more details, please see the following original paper:

Takeshi Saga, Hiroki Tanaka, and Satoshi Nakamura, "Computational analyses of linguistic features with schizophrenic and autistic traits along with formal thought disorders", Proceeding of ICMI, 2023.

Related URL:
- Paper: 
- Github: 