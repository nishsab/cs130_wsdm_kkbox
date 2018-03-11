import os
import gc
import datetime
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
import time

NOT_ENOUGH_THRESH=100
N_NEIGHBORS_LARGE=3
N_NEIGHBORS_SMALL=3
NUM_TRAINING_PTS=7500
def get_age(x):
    try:
        i = int(x)
        if i == 0:
            return "0"
        elif i < 13:
            return "OUTLIER"
        elif i <= 20:
            return "13-20"
        elif i <= 25:
            return "21-25"
        elif i <= 30:
            return "26-30"
        elif i <= 35:
            return "31-35"
        elif i <= 40:
            return "36-40"
        elif i <= 50:
            return "41-50"
        elif i <= 60:
            return "51-60"
        else:
            return "OUTLIER"
    except:
        return "OUTLIER"
AGES=["OUTLIER","0","13-20","21-25","26-30","31-35","36-40","41-50","51-60"]


SOURCE_SYSTEM_TAB='source_system_tab'
SOURCE_SCREEN_NAME='source_screen_name'
SOURCE_TYPE='source_type'
ARTIST_NAME='artist_name'
COMPOSER='composer'
LYRICIST='lyricist'
GENRE_IDS='genre_ids'
LANGUAGE='language'
REGISTERED_VIA='registered_via'
CITY='city'


csv_dir="test_csv"
OUTFILE='%s/test_out.csv' % (csv_dir)

try:
    os.remove(OUTFILE)
    print("deleting %s" % (OUTFILE))
except OSError:
    pass

print("hello")

def insert_or_add(d,k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1

def get_uniques(df,dict,col):
    for index, i in df.iterrows():
        insert_or_add(dict, i[col])

def get_genre(index, col):
    try:
        return col.split("|")[index]
    except:
        return "0"

def build_categorical(df,column_names,index_dict,large_training_set):
    rows = []
    for index, i in df.iterrows():
        #print(len(column_names))
        row = np.zeros((len(column_names),), dtype=int)
        #print(row)
        col = "%s_%s" % (SOURCE_SYSTEM_TAB, i[SOURCE_SYSTEM_TAB])
        if col in index_dict:
            row[index_dict[col]] = 1
        col = "%s_%s" % (SOURCE_SCREEN_NAME, i[SOURCE_SCREEN_NAME])
        if col in index_dict:
            row[index_dict[col]] = 1
        col = "%s_%s" % (SOURCE_TYPE, i[SOURCE_TYPE])
        if col in index_dict:
            row[index_dict[col]] = 1
        col = "%s_%s" % (ARTIST_NAME, i[ARTIST_NAME])
        if col in index_dict:
            row[index_dict[col]] = 1
        col = "%s_%s" % (LANGUAGE, i[LANGUAGE])
        if col in index_dict:
            row[index_dict[col]] = 1
        try:
            for c in i[COMPOSER].split("|"):
                col = "%s_%s" % (COMPOSER, c.strip())
                if col in index_dict:
                    row[index_dict[col]] = 1
        except:
            pass
        try:
            for l in i[LYRICIST].split("|"):
                col = "%s_%s" % (LYRICIST, l.strip())
                if col in index_dict:
                    row[index_dict[col]] = 1
        except:
            pass
        try:
            for g in i[GENRE_IDS].split("|"):
                col = "%s_%s" % (GENRE_IDS, g.strip())
                if col in index_dict:
                    row[index_dict[col]] = 1
        except:
            pass
        if not large_training_set:
            col = "%s_%s" % (CITY, i[CITY])
            if col in index_dict:
                row[index_dict[col]] = 1
            col = "%s_%s" % (REGISTERED_VIA, i[REGISTERED_VIA])
            if col in index_dict:
                row[index_dict[col]] = 1
            col = "%s_%s" % ("age", get_age(i['bd']))
            if col in index_dict:
                row[index_dict[col]] = 1
        rows.append(row)
    return pd.DataFrame(rows, columns=column_names)


def get_pds_from_csv():
    train = pd.read_csv('%s/minitrain.csv' % (csv_dir))
    member = pd.read_csv('%s/members.csv' % (csv_dir)).sort_values('msno')
    song = pd.read_csv('%s/songs.csv' % (csv_dir)).sort_values('song_id')
    train = train.merge(member[['msno', 'city', 'bd', 'gender', 'registered_via', \
                                'registration_init_time', 'expiration_date']], on='msno', how='left')

    train = train.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
                              'language', 'genre_ids']], on='song_id', how='left')
    train_f, test_f = train_test_split(train, test_size=0.2)
    test_f['id'] = test_f.index
    return train_f, test_f

def get_attributes(df,attribute_dict,large_training_set):
    for index, i in df.iterrows():
        insert_or_add(attribute_dict[SOURCE_SYSTEM_TAB], i[SOURCE_SYSTEM_TAB])
        insert_or_add(attribute_dict[SOURCE_SCREEN_NAME], i[SOURCE_SCREEN_NAME])
        insert_or_add(attribute_dict[SOURCE_TYPE], i[SOURCE_TYPE])
        insert_or_add(attribute_dict[ARTIST_NAME], i[ARTIST_NAME])
        insert_or_add(attribute_dict[LANGUAGE], i[LANGUAGE])
        try:
            for c in i[COMPOSER].split("|"):
                insert_or_add(attribute_dict[COMPOSER], c.strip())
        except:
            insert_or_add(attribute_dict[COMPOSER], "unknown")
        try:
            for l in i[LYRICIST].split("|"):
                insert_or_add(attribute_dict[LYRICIST], l.strip())
        except:
            insert_or_add(attribute_dict[LYRICIST], "unknown")
        try:
            for g in i[GENRE_IDS].split("|"):
                insert_or_add(attribute_dict[GENRE_IDS], g.strip())
        except:
            insert_or_add(attribute_dict[GENRE_IDS], "unknown")
        if not large_training_set:
            insert_or_add(attribute_dict[REGISTERED_VIA], i[REGISTERED_VIA])
            insert_or_add(attribute_dict[CITY], i[CITY])

def build_index(attributes,index_dict,column_names,large_training_set):
    i = 0
    for key in attributes:
        for value in attributes[key]:
            column_name = "%s_%s" % (key, value)
            column_names.append(column_name)
            index_dict[column_name] = i
            i += 1

    if not large_training_set:
        for age in AGES:
            column_names.append("age_%s" % (age))
            index_dict[column_name] = i
            i += 1

def get_attribute_dict(large_training_set):
    if large_training_set:
        return {SOURCE_SYSTEM_TAB: {}, SOURCE_SCREEN_NAME: {}, SOURCE_TYPE: {},
                                 ARTIST_NAME: {}, COMPOSER: {}, LYRICIST: {}, GENRE_IDS: {}, LANGUAGE: {}}
    else:
        return {SOURCE_SYSTEM_TAB: {}, SOURCE_SCREEN_NAME: {}, SOURCE_TYPE: {},
                                 ARTIST_NAME: {}, COMPOSER: {}, LYRICIST: {}, GENRE_IDS: {}, LANGUAGE: {},
                REGISTERED_VIA: {}, CITY: {}}


def main():
    print("Loading dataframes from csv files")
    start = int(round(time.time() * 1000))
    before = int(round(time.time() * 1000))
    train_f, test_f = get_pds_from_csv()
    users = {} #Users in test set
    not_enough_training = {} #Users with less than NOT_ENOUGH_THRESH training records
    get_uniques(test_f,users,'msno')

    after = int(round(time.time() * 1000))
    print("operation took %d ms" % (after - before))
    num_records_large=[]
    scores_large=[]

    progress=1
    total_users=len(users.keys())
    for user in users:
        train_local = train_f.loc[train_f['msno'] == user]
        num_training_records_for_user = len(train_local)

        print("(%d/%d): testing %s with %d training records" % (progress,total_users,user,num_training_records_for_user))
        progress+=1

        if num_training_records_for_user <= NOT_ENOUGH_THRESH:
            print("Adding %s to be processed later since it has less than %d training records" % (user,NOT_ENOUGH_THRESH))
            insert_or_add(not_enough_training, user)
            continue

        before = int(round(time.time() * 1000))
        train_y = train_local['target']

        large_training_attributes = get_attribute_dict(True)

        test_local = test_f.loc[test_f['msno'] == user]
        test_y = test_local['target']
        test_id = test_local['id']

        get_attributes(test_local,large_training_attributes,True)

        index_dict = {} #Map of column names to array index
        column_names = [] #List of column names

        build_index(large_training_attributes,index_dict,column_names,True)

        test_cat = build_categorical(test_local, column_names, index_dict,True)
        train_cat = build_categorical(train_local, column_names, index_dict,True)

        neigh = KNeighborsClassifier(n_neighbors=N_NEIGHBORS_LARGE)
        neigh.fit(train_cat, train_y)

        score = neigh.score(test_cat, test_y)

        num_records_large.append(len(test_cat))
        scores_large.append(score)

        try:
            out = np.column_stack((test_id, neigh.predict_proba(test_cat)[:, 1])) #Probability target is 1
        except:
            out = np.column_stack((test_id, neigh.predict_proba(test_cat))) #In case it's all 100%

        out_list = []
        for i in out:
            out_list.append("%f,%f" % (i[0], i[1]))

        f = open(OUTFILE, 'a')
        f.write("%s\n" % ("\n".join(out_list)))
        f.close()

        after = int(round(time.time() * 1000))
        print("Large Training Set Accuracy: User: %s, Test Records: %d, Accuracy: %f, Time: %d ms" % (user, len(test_y), score, after-before))

    train_small = train_f.sample(NUM_TRAINING_PTS)
    train_small_y = train_small['target']

    small_training_attributes = get_attribute_dict(False)
    get_attributes(train_small, small_training_attributes, False)

    small_index_dict = {}  # Map of column names to array index
    small_column_names = []  # List of column names
    build_index(small_training_attributes, small_index_dict, small_column_names, False)

    print("Training model for users with little training data")
    before = int(round(time.time() * 1000))

    train_cat_small = build_categorical(train_small, small_column_names, small_index_dict, False)
    neigh = KNeighborsClassifier(N_NEIGHBORS_SMALL)
    neigh.fit(train_cat_small, train_small_y)

    after = int(round(time.time() * 1000))
    print("operation took %d ms" % (after - before))

    num_records_small = []
    scores_small = []

    progress = 1
    total_small_users = len(not_enough_training.keys())
    for user in not_enough_training:

        test_local = test_f.loc[test_f['msno'] == user]
        test_y = test_local['target']
        test_id = test_local['id']

        print("(%d/%d): testing %s with %d records" % (progress, total_small_users, user, len(test_y)))
        progress += 1
        before = int(round(time.time() * 1000))

        test_cat = build_categorical(test_local, small_column_names, small_index_dict, False)
        score = neigh.score(test_cat, test_y)

        num_records_small.append(len(test_cat))
        scores_small.append(score)

        try:
            out = np.column_stack((test_id, neigh.predict_proba(test_cat)[:, 1])) #Probability target is 1
        except:
            out = np.column_stack((test_id, neigh.predict_proba(test_cat))) #In case it's all 100%

        out_list = []
        for i in out:
            out_list.append("%f,%f" % (i[0], i[1]))

        f = open(OUTFILE, 'a')
        f.write("%s\n" % ("\n".join(out_list)))
        f.close()

        after = int(round(time.time() * 1000))
        print("Small Training Set Accuracy: User: %s, Test Records: %d, Accuracy: %f, Time: %d ms" % (user, len(test_y), score, after-before))

    total_large = 0.
    runnning_total_large = 0
    for i in range(0, len(num_records_large)):
        total_large += num_records_large[i] * scores_large[i]
        runnning_total_large += num_records_large[i]

    total_small = 0.
    runnning_total_small = 0
    for i in range(0, len(num_records_small)):
        total_small += num_records_small[i] * scores_small[i]
        runnning_total_small += num_records_small[i]

    print("Large: %f" %(total_large / runnning_total_large))
    print("Small: %f" % (total_small / runnning_total_small))
    print("Total: %f" % ((total_large + total_small) / (runnning_total_large + runnning_total_small)))

    end = int(round(time.time() * 1000))
    print("Total Running Time: %d ms" % (end - start))

if __name__ == "__main__":
    main()
