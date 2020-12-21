import os
import pandas as pd
import collections
import numpy as np

num_counties = 401  # number of counties
num_govregions = 34  # number of local governing regions
abs_tol = 100  # maximum absolute error allowed per county migration
rel_tol = 0.01  # maximum relative error allowed per county migration

path = 'http://hpcagainstcorona.sc.bs.dlr.de/data/migration/'

counties = pd.read_excel(os.path.join(path, 'kreise_deu.xlsx'), sheet_name=1)


# print(counties)
# counties.info()

def get_data():
    # get all data generated in this file
    (countykey_list, countypop_list, govkey_list) = get_key_and_population_lists()
    (key2matindex, govkey2local) = create_hashmaps(countykey_list, govkey_list)
    (gov_table, key2govkey, key2localkey) = make_belonging_together_keys_list(countykey_list, govkey_list)
    state_gov_table = create_government_regions_list_per_state(govkey_list)
    mat_commuter_migration = get_matrix_commuter_migration_patterns(countypop_list, govkey_list, key2matindex,
                                                                    govkey2local,
                                                                    gov_table, key2govkey, key2localkey,
                                                                    state_gov_table)
    return (
        countykey_list, countypop_list, govkey_list, key2matindex, govkey2local, gov_table, key2govkey, key2localkey,
        state_gov_table, mat_commuter_migration)


def get_key_and_population_lists():
    # get and store all regional (county) identifiers in a list; store county populations accordingly
    # get a list of governing regions
    countykey_list = []
    countypop_list = []
    govkey_list = []
    for i in range(0, counties.shape[0]):
        # regional county identifieres (5 numbers)
        if (len(str(counties.iloc[i][0])) == 5 and (counties.iloc[i][0]).isdigit()):
            countykey_list.append(counties.iloc[i][0])
            countypop_list.append(counties.iloc[i][5])
            # print(counties.iloc[i][0], counties.iloc[i][2]) # print with county name

        # government region keys (2 or 3 numbers)
        elif (i < counties.shape[0] - 1 and len(str(counties.iloc[i][0])) < len(str(counties.iloc[i + 1][0]))):

            # workaround for old gov. regions and Saxony
            if (not str(counties.iloc[i][1]).startswith('früher') and not str(counties.iloc[i][1]).startswith(
                    'Direktion')):

                # only take those keys which have less numbers than the key in the next row
                if ((len(str(counties.iloc[i][0])) != 4 and len(str(counties.iloc[i + 1][0])) == 5)):
                    # where string length is not 4 and next key has length four
                    # these rows correspond to 'local government' regions (except for BW, RP and Saxony)
                    govkey_list.append(counties.iloc[i][0])
                    # print(counties.iloc[i][0], counties.iloc[i][1])

                elif (i < counties.shape[0] - 2):

                    if (len(str(counties.iloc[i][0])) == 3 and len(str(counties.iloc[i + 2][0])) == 5):
                        # workaround for BW; 'government regions' are again divided but do not appear as such in
                        # documents of the Arbeitsagentur
                        # print(counties.iloc[i][0], counties.iloc[i+2][0])
                        govkey_list.append(counties.iloc[i][0])
                        # print(counties.iloc[i][0], counties.iloc[i][1])

                    if (len(str(counties.iloc[i][0])) == 2 and len(str(counties.iloc[i + 2][0])) == 5):
                        # workaround for RP and Saxony;
                        if (str(counties.iloc[i + 1][1]).startswith('früher')):
                            # workaround for RP; 'government regions' were dissolved
                            govkey_list.append(counties.iloc[i][0])
                            # print(counties.iloc[i][0], counties.iloc[i][1], )

                        elif (str(counties.iloc[i + 1][1]).startswith('Direktion')):
                            # workaround for Saxony; 'Direktionsbezirke' not referred in commuter migration
                            govkey_list.append(counties.iloc[i][0])
                            # print(counties.iloc[i][0], counties.iloc[i][1], )

    if (len(govkey_list) != num_govregions):
        print('Error. Number of government regions wrong. Having', len(govkey_list), 'instead of',
              num_govregions)

    return (countykey_list, countypop_list, govkey_list)


def verify_sorted(countykey_list):
    # verify that read list is sorted
    sum_check = 0
    countykey_list_unique = np.unique(np.array(countykey_list))
    for i in range(0, len(countykey_list)):
        sum_check = int(countykey_list_unique[i]) - int(countykey_list[i])
        if (sum_check > 0):
            print('Error. Input list not sorted, population per county list had to be sorted accordingly.')


def create_hashmaps(countykey_list=None, govkey_list=None):
    # create a hashmap from sorted regional identifiers (01001 - ...) to 0 - num_counties
    if countykey_list is None or govkey_list is None:
        (countykey_list, help, govkey_list) = get_key_and_population_lists()
    verify_sorted(countykey_list)

    key2matindex = collections.OrderedDict()
    i = 0
    for index in countykey_list:
        key2matindex[index] = i
        i += 1

    if i != num_counties:
        print("Error. Number of counties wrong.")

    # create a hash map from sorted gov keys to local list
    govkey2local = collections.OrderedDict()
    i = 0
    for index in govkey_list:
        govkey2local[index] = i
        i += 1

    if i != num_govregions:
        print("Error. Number of governing regions wrong.")

    return (key2matindex, govkey2local)


def make_belonging_together_keys_list(countykey_list=None, govkey_list=None):
    # make list of government regions with lists of counties that belong to them
    # make list of states with government regions that belong to them
    # only works with sorted lists of keys
    if govkey_list is None or countykey_list is None:
        (countykey_list, help, govkey_list) = get_key_and_population_lists()

    verify_sorted(countykey_list)

    gov_table = []

    gov_index = 0
    col_index = 0
    col_list = []

    for i in range(0, len(countykey_list)):

        # check for belonging to currently considered government region
        if str(countykey_list[i]).startswith(str(govkey_list[gov_index])):
            col_list.append(countykey_list[i])  # add county to current government region
            col_index += 1
        # go to next government region
        if (i < len(countykey_list) - 1 and (not str(countykey_list[i + 1]).startswith(str(govkey_list[gov_index])))):
            gov_table.append(col_list)  # add government region to full table
            col_list = []
            gov_index += 1
            col_index = 0

    gov_table.append(col_list)  # add last government region

    if (len(gov_table) != num_govregions):
        print('Error. Number of government regions wrong.')

    # create a unique hash map from county key to its government region and a global key to local (in gov region) key ordering
    key2govkey = collections.OrderedDict()
    key2localkey = collections.OrderedDict()
    for i in range(0, len(gov_table)):
        for j in range(0, len(gov_table[i])):
            key2govkey[gov_table[i][j]] = i
            key2localkey[gov_table[i][j]] = j
    return (gov_table, key2govkey, key2localkey)


def create_government_regions_list_per_state(govkey_list=get_key_and_population_lists()[2]):
    # create government regions list per state
    state_gov_table = []

    state_id = 1
    state_govlist_loc = []
    for i in range(0, len(govkey_list)):

        if (str(int(govkey_list[i])).startswith(str(state_id))):
            state_govlist_loc.append(govkey_list[i])

        if (i + 1 < len(govkey_list) and not str(int(govkey_list[i + 1])).startswith(str(state_id))):
            state_id += 1
            state_gov_table.append(state_govlist_loc)
            state_govlist_loc = []

    state_gov_table.append(state_govlist_loc)  # add last state's list
    return state_gov_table


def get_matrix_commuter_migration_patterns(countypop_list=None, govkey_list=None, key2matindex=None, govkey2local=None,
                                           gov_table=None, key2govkey=None, key2localkey=None, state_gov_table=None):
    # matrix of commuter migration patterns
    if countypop_list is None or govkey_list is None:
        (countykey_list, countypop_list, govkey_list) = get_key_and_population_lists()
    if key2matindex is None or govkey2local is None:
        (key2matindex, govkey2local) = create_hashmaps(countykey_list, govkey_list)
    if gov_table is None or key2govkey is None or key2localkey is None:
        (gov_table, key2govkey, key2localkey) = make_belonging_together_keys_list(countykey_list, govkey_list)
    if state_gov_table is None:
        state_gov_table = create_government_regions_list_per_state(govkey_list)

    mat_commuter_migration = np.zeros((num_counties, num_counties))

    # maxium errors (of people not detected)
    max_abs_err = 0
    max_rel_err = 0

    files = []
    for n in range(1, 10):
        files.append('krpend_0' + str(n) + "_0.xlsx")
    for n in range(10, 17):
        files.append('krpend_' + str(n) + "_0.xlsx")

    n = 0
    for item in files:
        # Using the 'Einpendler' sheet to correctly distribute summed values over counties of other gov. region
        commuter_migration_file = pd.read_excel(os.path.join(path, item), sheet_name=3)
        # commuter_migration_file.info()

        counties_done = []  # counties considered as 'migration from'
        current_row = -1  # row of matrix that belongs to county migrated from
        current_col = -1  # column of matrix that belongs to county migrated to
        checksum = 0  # sum of county migration from, to be checked against sum in document

        for i in range(0, commuter_migration_file.shape[0]):

            # print(commuter_migration_file.iloc[i][1])
            # if(str(commuter_migration_file.iloc[i][0]).startswith('03354')):

            if (len(str(commuter_migration_file.iloc[i][0])) == 5
                    and (commuter_migration_file.iloc[i][0]).isdigit()):
                checksum = 0
                # make zero'd list of counties explicitly migrated to from county considered
                # 'implicit' migration means 'migration to' which is summed in a larger regional entity and not given in detail per county
                counties_migratedfrom = []
                for j in range(0, len(gov_table)):
                    counties_migratedfrom.append(np.zeros(len(gov_table[j])))

                counties_done.append(commuter_migration_file.iloc[i][0])
                current_col = key2matindex[commuter_migration_file.iloc[i][0]]
                curr_county_migratedto = commuter_migration_file.iloc[i][1]
                current_key = commuter_migration_file.iloc[i][0]
                current_name = commuter_migration_file.iloc[i][1]
                # migration to itself excluded!
                counties_migratedfrom[key2govkey[current_key]][key2localkey[current_key]] = 1

            if (type(commuter_migration_file.iloc[i][
                         2]) != float):  # removal of nan's, regional keys are stored as strings

                if ((commuter_migration_file.iloc[i][2]).isdigit()):  # check if entry is a digit
                    # print(commuter_migration_file.iloc[i][0], commuter_migration_file.iloc[i][2], type(commuter_migration_file.iloc[i][2]))
                    # print((commuter_migration_file.iloc[i][2]).isdigit(), float(commuter_migration_file.iloc[i-1][2]), str(commuter_migration_file.iloc[i-1][2]).startswith('nan'))
                    # explicit migration from county to county
                    if (len(str(commuter_migration_file.iloc[i][
                                    2])) == 5):  # check if entry refers to a specific county, then set matrix value
                        current_row = key2matindex[commuter_migration_file.iloc[i][2]]
                        val = commuter_migration_file.iloc[i][4]
                        mat_commuter_migration[current_row, current_col] = val
                        checksum += val
                        # print(val)
                        counties_migratedfrom[key2govkey[commuter_migration_file.iloc[i][2]]][
                            key2localkey[commuter_migration_file.iloc[i][2]]] = 1
                        # print(current_row, current_col, val)

                    # take summed values of other REMAINING counties of government region
                    # here, some counties of the region are stated explicitly and the rest is summed
                    elif (str(commuter_migration_file.iloc[i][3]) == 'Übrige Kreise (Regierungsbezirk)' and str(
                            commuter_migration_file.iloc[i][4]).isdigit()):

                        # remove trailing zeros (dummy key w/o zeros: dummy_key_wozeros)
                        dummy_key_wozeros = str(commuter_migration_file.iloc[i][2])
                        if (len(dummy_key_wozeros) > 2 and dummy_key_wozeros[2] == '0'):
                            dummy_key_wozeros = dummy_key_wozeros[0:2]

                            # sum population of all counties not explicitly migrated from of the current gov region migrated from
                        dummy_pop_sum = 0
                        for k in range(0, len(gov_table[govkey2local[dummy_key_wozeros]])):
                            if (counties_migratedfrom[govkey2local[dummy_key_wozeros]][k] < 1):
                                # get identifier (0-401) for county key
                                globindex = key2matindex[gov_table[govkey2local[dummy_key_wozeros]][k]]
                                # sum up
                                dummy_pop_sum += countypop_list[globindex]

                        # distribute emigration relatively to county population where migration comes from
                        # dummy_checksum = 0
                        for k in range(0, len(gov_table[govkey2local[dummy_key_wozeros]])):
                            if (counties_migratedfrom[govkey2local[dummy_key_wozeros]][k] < 1):
                                # get identifier (0-401) for county key
                                globindex = key2matindex[gov_table[govkey2local[dummy_key_wozeros]][k]]
                                counties_migratedfrom[govkey2local[dummy_key_wozeros]][k] = 1

                                # set value computed relatively to county size and effective migration
                                current_row = globindex
                                val = commuter_migration_file.iloc[i][4] * countypop_list[globindex] / dummy_pop_sum
                                checksum += val
                                # dummy_checksum += val
                                mat_commuter_migration[current_row, current_col] = val
                        # print(dummy_checksum)

                    # take summed values of ALL counties of a government region
                    # here, no single county of the region is stated explicitly, all counties are summed together
                    elif (commuter_migration_file.iloc[i][2] in govkey_list and sum(
                            counties_migratedfrom[govkey2local[commuter_migration_file.iloc[i][2]]]) == 0):

                        # sum population of all counties not explicitly migrated to of the current gov region migrated to
                        dummy_pop_sum = 0
                        for k in range(0, len(gov_table[govkey2local[commuter_migration_file.iloc[i][2]]])):
                            if (counties_migratedfrom[govkey2local[commuter_migration_file.iloc[i][2]]][k] < 1):
                                # get identifier (0-401) for county key
                                globindex = key2matindex[gov_table[govkey2local[commuter_migration_file.iloc[i][2]]][k]]
                                # sum up
                                dummy_pop_sum += countypop_list[globindex]

                        # distribute emigration relatively to county population where migration comes from
                        # dummy_checksum = 0
                        for k in range(0, len(gov_table[govkey2local[commuter_migration_file.iloc[i][2]]])):
                            if (counties_migratedfrom[govkey2local[commuter_migration_file.iloc[i][2]]][k] < 1):
                                # get identifier (0-401) for county key
                                globindex = key2matindex[gov_table[govkey2local[commuter_migration_file.iloc[i][2]]][k]]
                                counties_migratedfrom[govkey2local[commuter_migration_file.iloc[i][2]]][k] = 1

                                # set value computed relatively to county size and effective migration
                                current_row = globindex
                                val = commuter_migration_file.iloc[i][4] * countypop_list[globindex] / dummy_pop_sum
                                checksum += val
                                # dummy_checksum += val
                                mat_commuter_migration[current_row, current_col] = val
                        # print(dummy_checksum)

                    # take summed values of other REMAINING counties of a whole Bundesland
                    # here, some counties of the Bundesland are stated explicitly and the rest is summed
                    # the first or is for the case that the right first line of the incoming people directly
                    # addresses one
                    # the latter 'or's is used if no single county nor gov region of a federal state is stated explicitly
                    # although there are existent government regions in this federal state (i.e., the state itself is not
                    # considered a governement region according to gov_list)
                    elif ((str(commuter_migration_file.iloc[i][3]) == 'Übrige Regierungsbezirke (Bundesland)' and str(
                            commuter_migration_file.iloc[i][4]).isdigit())
                          or ((commuter_migration_file.iloc[i][2]).isdigit() and str(
                                commuter_migration_file.iloc[i - 1][2]).startswith('nan'))
                          or (len(str(commuter_migration_file.iloc[i][2])) == 2 and
                              abs(float(commuter_migration_file.iloc[i][2]) - float(
                                  commuter_migration_file.iloc[i - 1][2])) == 1)
                          or (len(str(commuter_migration_file.iloc[i][2])) == 2 and
                              abs(float(commuter_migration_file.iloc[i][2]) - float(
                                  commuter_migration_file.iloc[i - 1][2])) == 2)):

                        # auxiliary key of Bundesland (key translated to int starting at zero)
                        dummy_key = int(commuter_migration_file.iloc[i][2]) - 1

                        # sum population of all counties not explicitly migrated from the current gov region migrated from
                        dummy_pop_sum = 0
                        for j in range(0, len(
                                state_gov_table[dummy_key])):  # over all government regions not explicitly stated
                            gov_index = govkey2local[state_gov_table[dummy_key][j]]
                            for k in range(0,
                                           len(gov_table[gov_index])):  # over all counties of the considered gov region
                                if (counties_migratedfrom[gov_index][k] < 1):
                                    # get identifier (0-401) for county key
                                    globindex = key2matindex[gov_table[gov_index][k]]
                                    # sum up
                                    dummy_pop_sum += countypop_list[globindex]

                        # distribute emigration relatively to county population where migration comes from
                        # dummy_checksum = 0
                        for j in range(0, len(
                                state_gov_table[dummy_key])):  # over all government regions not explicitly stated
                            gov_index = govkey2local[state_gov_table[dummy_key][j]]
                            for k in range(0,
                                           len(gov_table[gov_index])):  # over all counties of the considered gov region
                                if (counties_migratedfrom[gov_index][k] < 1):
                                    # get identifier (0-401) for county key
                                    globindex = key2matindex[gov_table[gov_index][k]]
                                    counties_migratedfrom[gov_index][k] = 1

                                    # set value computed relatively to county size and effective migration
                                    current_row = globindex
                                    val = commuter_migration_file.iloc[i][4] * countypop_list[globindex] / dummy_pop_sum
                                    checksum += val
                                    # dummy_checksum += val
                                    mat_commuter_migration[current_row, current_col] = val
                                    # print(countypop_list[globindex], dummy_pop_sum, val)

                        # print(dummy_checksum)

            # sum of total migration 'from'
            if (str(commuter_migration_file.iloc[i][3]) == 'Einpendler aus dem Bundesgebiet'):
                abs_err = abs(checksum - commuter_migration_file.iloc[i][4])
                if (abs_err > max_abs_err):
                    max_abs_err = abs_err
                if (abs_err / checksum > max_rel_err):
                    max_rel_err = abs_err / checksum
                if (abs_err < abs_tol and abs_err / checksum < rel_tol):
                    # print('Absolute error:', abs_err, '\t relative error:', abs_err/checksum)
                    checksum = 0
                else:
                    print('Error in calculations for county ', curr_county_migratedto,
                          '\nAccumulated values:', checksum,
                          ', correct sum:', commuter_migration_file.iloc[i][4])
                    print('Absolute error:', abs_err, ', relative error:', abs_err / checksum)
                    # break

        n += 1
        print('Federal state read. Progress ', n, '/ 16')
    if n != 16:
        print('Error. Files missing.')

    print('Maximum absolute error:', max_abs_err)
    print('Maximum relative error:', max_rel_err)
    testing(key2matindex, mat_commuter_migration, countypop_list)
    return mat_commuter_migration


def testing(key2matindex, mat_commuter_migration, countypop_list):
    # just do some tests on randomly chosen migrations

    # check migration from Leverkusen (averaged from NRW, 05) to Hildburghausen
    city_from = key2matindex['05316']
    city_to = key2matindex['16069']
    if (countypop_list[city_from] != 163729 or mat_commuter_migration[city_from][city_to] != 34 * countypop_list[
        city_from] / 17947221):
        print(countypop_list[city_from], mat_commuter_migration[city_to][city_from])
        print('Error')

    # check migration from Duisburg to Oberspreewald-Lausitz
    city_from = key2matindex['05112']
    city_to = key2matindex['12066']
    if (mat_commuter_migration[city_from][city_to] != 10):
        print('Error')

    # check migration from Lahn-Dill-Kreis to Hamburg
    city_from = key2matindex['06532']
    city_to = key2matindex['02000']
    if (mat_commuter_migration[city_from][city_to] != 92):
        print('Error')

        # check migration from Landsberg am Lech (averaged from 091) to Hersfeld-Rotenburg
    city_from = key2matindex['09181']
    city_to = key2matindex['06632']
    if (mat_commuter_migration[city_from][city_to] != 47 * 120302 / (4710865 - 1484226)):
        print('Error')

    # check migration from Herzogtum Lauenburg to Flensburg, Stadt
    city_from = key2matindex['01001']
    city_to = key2matindex['01053']
    if (mat_commuter_migration[city_from][city_to] != 17):
        print('Error')


if __name__ == "__main__":
    get_data()
