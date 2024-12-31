#!/usr/bin/python
# Run Tamarin interactive mode and graphcrawler for all .spthy 
# automatically.

import subprocess, os, sys, shutil, time
import collect
import logging

# config log
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(filename='rescrawler.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def tmp_copy(spthy):
    """ Copy the .spthy to ./tmp/ folder (relative to itself)
    Para:
        spthy: path to .spthy (relative to this script)
    Return:
        New path to .spthy
    """
    spthy_folder = os.path.dirname(spthy)
    tmp_folder = spthy_folder + '/tmp/'

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    os.mkdir(tmp_folder)

    tmp_spthy = shutil.copy(spthy, tmp_folder)

    with open(tmp_spthy, "r", encoding="utf8") as f:
        tmp_content = f.read()

    tmp_content = tmp_content.replace("restriction Atomic:", 
                                      """
rule (modulo AC) d_0_rm:
    [ !KD( cmac(mo + mk, k) ), !KU( mo ), !KU( k ) ]
    --[
        Atomic(mk)
    ]->
    [ !KD( mo + mk ) ]

rule (modulo AC) d_1_rm:
    [ !KD( cmac(m, k) ), !KU( k ) ]
    --[
        Atomic(m)
    ]->
    [ !KD( m ) ]

rule (modulo AC) d_0_0_rm_cmac:
    [ !KD( mo + mk ), !KU( mo ), !KU( k ), !KU( k ) ]
    --[
        Atomic(mk)
    ]->
    [ !KD( mo + mk ) ]

rule (modulo AC) d_0_pair:
    [ !KD( fst(x) ), !KU( snd(x) ) ]
    --[
        Neq(<fst(x),snd(x)>, x)
    ]->
    [ !KD( x ) ]

rule (modulo AC) d_1_pair:
    [ !KD( snd(x) ), !KU( fst(x) ) ]
    --[
        Neq(<fst(x),snd(x)>, x)
    ]->
    [ !KD( x ) ]

restriction Atomic:""")
    
    with open(tmp_spthy, "w", encoding="utf8") as f:
        f.write(tmp_content)

    return tmp_spthy


def tmp_delete(tmp_spthy):
    """ Delete /tmp/
    Para:
        tmp_spthy: .spthy under tmp folder
        spthy: the original path of .spthy
    Return:
        Nothing.
    """
    tmp_folder = os.path.dirname(tmp_spthy)
    shutil.rmtree(tmp_folder)

def main():
    try:
        res_folder = sys.argv[1]
    except IndexError:
        logging.warning("Not assign which folder are the results saved. Ask input folder name.")
        res_folder = input("Under which folder are the results saved? (relative to this script)\n")
        logging.info("folder " + res_folder + " has been input.")

    logging.info("Starting collect spthy files...")
    spthy_list = collect.get_spthy_paths(res_folder)

    print("{} .spthy in total".format(len(spthy_list)))
    logging.info("Finished collect spthy files. {} .spthy in total".format(len(spthy_list)))

    logging.info("Begin enumerate spthy file list...")
    for index, spthy in enumerate(spthy_list):
        print("=========================================")
        print("{} / {}: ".format(index+1, len(spthy_list)), end="")
        print(spthy)
        logging.info("Handle {} / {}: {}".format(index+1, len(spthy_list), spthy))
        tmp_spthy = tmp_copy(spthy)

        print("\nRun tamarin interactive mode")
        logging.info("\tRun tamarin interactive mode")
        p_tamarin = subprocess.Popen(["nohup","tamarin-prover","interactive","--image-format=SVG",tmp_spthy],shell=False)

        try:
            p_tamarin.wait(20)
        except subprocess.TimeoutExpired:
            print("\nHaving waited for 20 seconds.")

        print("Begin crawling...")
        logging.info("\tBegin crawling...")
        try:
            p_crawler = subprocess.Popen(["python3","./graphcrawler.py",tmp_spthy],shell=False)
            p_crawler.wait()
        except BaseException as e:
            logging.info("\tMeet Exception during running graphcrawler.py: " + e)

        print("Finish crawling for " + spthy)
        print('\n\n')
        logging.info("\tFinish crawling for " + spthy)

        try:
            p_tamarin.terminate()
        except:
            logging.info("\tException during terminate tamarin interactive thread")

        os.remove('nohup.out')

        tmp_delete(tmp_spthy)
    logging.info("End enumerate spthy file list")


if __name__ == "__main__":
    try:
        main()
    finally:
        t = time.strftime('%Y%m%d_%H%M', time.localtime())
        os.rename("./rescrawler.log", "./crawlog_{}.txt".format(t))