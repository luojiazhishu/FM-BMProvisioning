SEPARATOR = ==============================================================================
# File = PROV_TEMP.m4
File = FIXUP_TEMP.m4
# Dir = Results/Unfixed
Dir = Results/Fixed
INBAND_OUTDir = $(Dir)/INBAND_OUT
INBAND_INDir = $(Dir)/INBAND_IN
INBAND_STATICDir = $(Dir)/INBAND_STATIC
INBAND_NOOOBDir = $(Dir)/INBAND_NOOOB
OOBPK_OUTDir = $(Dir)/OOBPK_OUT
OOBPK_INDir = $(Dir)/OOBPK_IN
OOBPK_STATICDir = $(Dir)/OOBPK_STATIC
OOBPK_NOOOBDir = $(Dir)/OOBPK_NOOOB
Threads = +RTS -N60 -RTS
Heuristic_S = --heuristic=S
Diff = --diff

Lemma = *

ifeq ($(Lemma),*)
    fnLemma=All
else 
	fnLemma=$(Lemma)
endif


dnl ################################################
dnl Use ARGSDefine to model different device type pairs.
dnl the meaning of the position is listed above,
dnl 1: ProvReadStaticOOB; 2: ProvReadOOBPK; 3: NDevPKType;
dnl 4: NDevOutputOOB; 5:NDevStaticOOB; 6:NDevStaticOOB
divert(-1)
changequote(<!,!>)
define(ARGSDefine,
<!-DARGPStaticOOB=$1 -DARGPOOBPK=$2 -DARGDPKType=$3 -DARGDOutputOOB=$4 -DARGDInputOOB=$5 -DARGDStaticOOB=$6!>)
changequote
divert(0)dnl


dnl "A general template for generating .sphty file and prove it."
dnl "Can be used for defined customized macro for different paths."
divert(-1)
changequote(<!,!>)
define(Generate_spthy,
<!m4 ARGSDefine($1,$2,$3,$4,$5,$6) $(File) > ./$($7)/MESH_$1$2$3$4$5$6.spthy!>)
define(TamarinProve,
<!tamarin-prover $(Heuristic_S) --prove=$(Lemma) ./$($7)/MESH_$1$2$3$4$5$6.spthy --output=./$($7)/proofs/Out_$(fnLemma)_MESH_$1$2$3$4$5$6.spthy > .tmp
	echo >> ./$($7)/proofs/Out_$(fnLemma)_MESH_$1$2$3$4$5$6.spthy
	cat .tmp >> ./$($7)/proofs/Out_$(fnLemma)_MESH_$1$2$3$4$5$6.spthy !>)
changequote
divert(0)dnl


dnl TODO: rewrite test using the template macro
test:
	m4 ARGSDefine(0,1,OOB,0,1,1) $(File) > ./test.spthy
	tamarin-prover $(Heuristic_S) --prove=$(Lemma) ./test.spthy --output=./out_test.spthy

testclean:
	rm test.spthy out_test.spthy

ALL: INBAND_OUT INBAND_IN INBAND_STATIC INBAND_NOOOB OOBPK_OUT OOBPK_IN OOBPK_STATIC OOBPK_NOOOB

INBAND: INBAND_OUT INBAND_IN INBAND_STATIC INBAND_NOOOB

OOBPK: OOBPK_OUT OOBPK_IN OOBPK_STATIC OOBPK_NOOOB

OUTPUT: INBAND_OUT OOBPK_OUT

INPUT: INBAND_IN OOBPK_IN

STATIC: INBAND_STATIC OOBPK_STATIC

NOOOB: INBAND_NOOOB INBAND_NOOOB

dnl ###################################################
INBAND_OUT:
	mkdir -p $(INBAND_OUTDir)
	mkdir -p $(INBAND_OUTDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(INBAND_OUT_Generate_spthy,
<!Generate_spthy($1,$2,$3,1,$4,$5,INBAND_OUTDir)!>)
define(INBAND_OUT_TamarinProve,
<!TamarinProve($1,$2,$3,1,$4,$5,INBAND_OUTDir)!>)
define(INBAND_OUT_Generate_Prove,
<!INBAND_OUT_Generate_spthy($1,$2,$3,$4,$5)
	INBAND_OUT_TamarinProve($1,$2,$3,$4,$5)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl INBAND_OUT_Generate_Prove(PStaticOOB,POOBPK,DPKType,
dnl     DOutputOOB = 1, DInputOOB, DStaticOOB)
	INBAND_OUT_Generate_Prove(0,0,InBand,0,0)
	INBAND_OUT_Generate_Prove(0,0,InBand,0,1)
	INBAND_OUT_Generate_Prove(1,0,InBand,0,0)
	INBAND_OUT_Generate_Prove(0,0,OOB,0,0)
	INBAND_OUT_Generate_Prove(0,0,OOB,0,1)
	INBAND_OUT_Generate_Prove(1,0,OOB,0,0)
	INBAND_OUT_Generate_Prove(0,1,InBand,0,0)
	INBAND_OUT_Generate_Prove(0,1,InBand,0,1)
	INBAND_OUT_Generate_Prove(1,1,InBand,0,0)
	INBAND_OUT_Generate_Prove(0,0,InBand,1,0)
	INBAND_OUT_Generate_Prove(0,0,InBand,1,1)
	INBAND_OUT_Generate_Prove(1,0,InBand,1,0)
	INBAND_OUT_Generate_Prove(0,0,OOB,1,0)
	INBAND_OUT_Generate_Prove(0,0,OOB,1,1)
	INBAND_OUT_Generate_Prove(1,0,OOB,1,0)
	INBAND_OUT_Generate_Prove(0,1,InBand,1,0)
	INBAND_OUT_Generate_Prove(0,1,InBand,1,1)
	INBAND_OUT_Generate_Prove(1,1,InBand,1,0)


dnl ###################################################
INBAND_IN:
	mkdir -p $(INBAND_INDir)
	mkdir -p $(INBAND_INDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(INBAND_IN_Generate_spthy,
<!Generate_spthy($1,$2,$3,0,1,$4,INBAND_INDir)!>)
define(INBAND_IN_TamarinProve,
<!TamarinProve($1,$2,$3,0,1,$4,INBAND_INDir)!>)
define(INBAND_IN_Generate_Prove,
<!INBAND_IN_Generate_spthy($1,$2,$3,$4)
	INBAND_IN_TamarinProve($1,$2,$3,$4)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl INBAND_IN_Generate_Prove(PStaticOOB,POOBPK,DPKType,
dnl     DOutputOOB = 0, DInputOOB = 1, DStaticOOB)
	INBAND_IN_Generate_Prove(0,0,InBand,0)
	INBAND_IN_Generate_Prove(0,0,InBand,1)
	INBAND_IN_Generate_Prove(1,0,InBand,0)
	INBAND_IN_Generate_Prove(0,0,OOB,0)
	INBAND_IN_Generate_Prove(0,0,OOB,1)
	INBAND_IN_Generate_Prove(1,0,OOB,0)
	INBAND_IN_Generate_Prove(0,1,InBand,0)
	INBAND_IN_Generate_Prove(0,1,InBand,1)
	INBAND_IN_Generate_Prove(1,1,InBand,0)

dnl ###################################################
INBAND_STATIC:
	mkdir -p $(INBAND_STATICDir)
	mkdir -p $(INBAND_STATICDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(INBAND_STATIC_Generate_spthy,
<!Generate_spthy(1,$1,$2,$3,$4,1,INBAND_STATICDir)!>)
define(INBAND_STATIC_TamarinProve,
<!TamarinProve(1,$1,$2,$3,$4,1,INBAND_STATICDir)!>)
define(INBAND_STATIC_Generate_Prove,
<!INBAND_STATIC_Generate_spthy($1,$2,$3,$4)
	INBAND_STATIC_TamarinProve($1,$2,$3,$4)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl INBAND_IN_Generate_Prove(PStaticOOB = 1,POOBPK,DPKType,
dnl     DOutputOOB, DInputOOB, DStaticOOB = 1)
	INBAND_STATIC_Generate_Prove(0,InBand,0,0)
	INBAND_STATIC_Generate_Prove(0,InBand,0,1)
	INBAND_STATIC_Generate_Prove(0,InBand,1,0)
	INBAND_STATIC_Generate_Prove(0,InBand,1,1)
	INBAND_STATIC_Generate_Prove(0,OOB,0,0)
	INBAND_STATIC_Generate_Prove(0,OOB,0,1)
	INBAND_STATIC_Generate_Prove(0,OOB,1,0)
	INBAND_STATIC_Generate_Prove(0,OOB,1,1)
	INBAND_STATIC_Generate_Prove(1,InBand,0,0)
	INBAND_STATIC_Generate_Prove(1,InBand,0,1)
	INBAND_STATIC_Generate_Prove(1,InBand,1,0)
	INBAND_STATIC_Generate_Prove(1,InBand,1,1)

dnl ###################################################
INBAND_NOOOB:
	mkdir -p $(INBAND_NOOOBDir)
	mkdir -p $(INBAND_NOOOBDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(INBAND_NOOOB_Generate_spthy,
<!Generate_spthy($1,$2,$3,0,0,$4,INBAND_NOOOBDir)!>)
define(INBAND_NOOOB_TamarinProve,
<!TamarinProve($1,$2,$3,0,0,$4,INBAND_NOOOBDir)!>)
define(INBAND_NOOOB_Generate_Prove,
<!INBAND_NOOOB_Generate_spthy($1,$2,$3,$4)
	INBAND_NOOOB_TamarinProve($1,$2,$3,$4)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl INBAND_IN_Generate_Prove(PStaticOOB,POOBPK,DPKType,
dnl     DOutputOOB = 0, DInputOOB = 0, DStaticOOB)
	INBAND_NOOOB_Generate_Prove(0,0,InBand,0)
	INBAND_NOOOB_Generate_Prove(0,0,InBand,1)
	INBAND_NOOOB_Generate_Prove(1,0,InBand,0)
	INBAND_NOOOB_Generate_Prove(0,0,OOB,0)
	INBAND_NOOOB_Generate_Prove(0,0,OOB,1)
	INBAND_NOOOB_Generate_Prove(1,0,OOB,0)
	INBAND_NOOOB_Generate_Prove(0,1,InBand,0)
	INBAND_NOOOB_Generate_Prove(0,1,InBand,1)
	INBAND_NOOOB_Generate_Prove(1,1,InBand,0)


dnl ###################################################
OOBPK_OUT:
	mkdir -p $(OOBPK_OUTDir)
	mkdir -p $(OOBPK_OUTDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(OOBPK_OUT_Generate_spthy,
<!Generate_spthy($1,1,OOB,1,$2,$3,OOBPK_OUTDir)!>)
define(OOBPK_OUT_TamarinProve,
<!TamarinProve($1,1,OOB,1,$2,$3,OOBPK_OUTDir)!>)
define(OOBPK_OUT_Generate_Prove,
<!OOBPK_OUT_Generate_spthy($1,$2,$3)
	OOBPK_OUT_TamarinProve($1,$2,$3)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl OOBPK_OUT_Generate_Prove(PStaticOOB,POOBPK = 1,DPKType = OOB,
dnl     DOutputOOB = 1, DInputOOB, DStaticOOB)
	OOBPK_OUT_Generate_Prove(0,0,0)
	OOBPK_OUT_Generate_Prove(0,0,1)
	OOBPK_OUT_Generate_Prove(1,0,0)
	OOBPK_OUT_Generate_Prove(0,1,0)
	OOBPK_OUT_Generate_Prove(0,1,1)
	OOBPK_OUT_Generate_Prove(1,1,0)

dnl ###################################################
OOBPK_IN:
	mkdir -p $(OOBPK_INDir)
	mkdir -p $(OOBPK_INDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(OOBPK_IN_Generate_spthy,
<!Generate_spthy($1,1,OOB,0,1,$2,OOBPK_INDir)!>)
define(OOBPK_IN_TamarinProve,
<!TamarinProve($1,1,OOB,0,1,$2,OOBPK_INDir)!>)
define(OOBPK_IN_Generate_Prove,
<!OOBPK_IN_Generate_spthy($1,$2)
	OOBPK_IN_TamarinProve($1,$2)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl OOBPK_IN_Generate_Prove(PStaticOOB,POOBPK = 1,DPKType = OOB,
dnl     DOutputOOB = 0, DInputOOB = 1, DStaticOOB)
	OOBPK_IN_Generate_Prove(0,0)
	OOBPK_IN_Generate_Prove(0,1)
	OOBPK_IN_Generate_Prove(1,0)

dnl ###################################################
OOBPK_STATIC:
	mkdir -p $(OOBPK_STATICDir)
	mkdir -p $(OOBPK_STATICDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(OOBPK_STATIC_Generate_spthy,
<!Generate_spthy(1,1,OOB,$1,$2,1,OOBPK_STATICDir)!>)
define(OOBPK_STATIC_TamarinProve,
<!TamarinProve(1,1,OOB,$1,$2,1,OOBPK_STATICDir)!>)
define(OOBPK_STATIC_Generate_Prove,
<!OOBPK_STATIC_Generate_spthy($1,$2)
	OOBPK_STATIC_TamarinProve($1,$2)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl OOBPK_STATIC_Generate_Prove(PStaticOOB = 1,POOBPK = 1,DPKType = OOB,
dnl     DOutputOOB, DInputOOB, DStaticOOB = 1)
	OOBPK_STATIC_Generate_Prove(0,0)
	OOBPK_STATIC_Generate_Prove(0,1)
	OOBPK_STATIC_Generate_Prove(1,0)
	OOBPK_STATIC_Generate_Prove(1,1)

dnl ###################################################
OOBPK_NOOOB:
	mkdir -p $(OOBPK_NOOOBDir)
	mkdir -p $(OOBPK_NOOOBDir)/proofs
dnl ###################################################
divert(-1)
changequote(<!,!>)
define(OOBPK_NOOOB_Generate_spthy,
<!Generate_spthy($1,1,OOB,0,0,$2,OOBPK_NOOOBDir)!>)
define(OOBPK_NOOOB_TamarinProve,
<!TamarinProve($1,1,OOB,0,0,$2,OOBPK_NOOOBDir)!>)
define(OOBPK_NOOOB_Generate_Prove,
<!OOBPK_NOOOB_Generate_spthy($1,$2)
	OOBPK_NOOOB_TamarinProve($1,$2)!>)
changequote
divert(0)dnl
dnl ###################################################
dnl OOBPK_NOOOB_Generate_Prove(PStaticOOB,POOBPK = 1,DPKType = OOB,
dnl     DOutputOOB = 0, DInputOOB = 0, DStaticOOB)
	OOBPK_NOOOB_Generate_Prove(0,0)
	OOBPK_NOOOB_Generate_Prove(0,1)
	OOBPK_NOOOB_Generate_Prove(1,0)
