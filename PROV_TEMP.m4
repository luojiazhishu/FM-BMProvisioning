theory PROVISIONING
begin
builtins: diffie-hellman, symmetric-encryption, multiset
functions: s1/1,cmac/2,rm/3,rm1/2

/*
We define the message authentication code function AES-CMAC
using the above equations and function symbols.
           tag = cmac(m1,m2,k)
           vrfy(tag,m1,m2,k) = true
           rm1(tag,m2,k) = m1
           rm2(tag,m1,k) = m2

             ┌───┐      ┌───┐
             │M_1│      │M_2│
             └─┬─┘      └─┬─┘
               ↓          ↓
              XOR   ┌─-->XOR<-─K_1
               ↓    │     ↓ 
            ┌─────┐ │  ┌─────┐
            │AES_k│ │  │AES_k│
            └──┬──┘ │  └──┬──┘
               └────┘     ↓
                         tag
                 AES-CMAC
*/

// Use multiset to model AES-CMAC
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

restriction Atomic:
    "All mk #i. Atomic(mk) @#i ==>
        ( not (Ex x y. x + y = mk) )
    "

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

/*************
*  Channels  *
*************/
// OOB channel for PK, secure for now
rule OOBPK_OutChannel [color=#9AFF9A]:
    [ Out_OOBPK(<channelname,SendType,ReceiveType>,D,m)]
    -->
    [ SecureOOBPK(<channelname,SendType,ReceiveType>,D,m)]
rule OOBPK_InChannel:
    [ SecureOOBPK(<channelname,SendType,ReceiveType>,D,m)]
    -->
    [ In_OOBPK(<channelname,SendType,ReceiveType>,D,m)]

// OOB channel for static oob info
rule StaticOOB_OutChannel [color=#9AFF9A]:
    [ Out_StaticOOB(<channelname,SendType,ReceiveType>,D,m)]
    -->
    [ SecureStaticOOB(<channelname,SendType,ReceiveType>,D,m)]
rule StaticOOB_InChannel:
    [ SecureStaticOOB(<channelname,SendType,ReceiveType>,D,m)]
    -->
    [ In_StaticOOB(<channelname,SendType,ReceiveType>,D,m)]

// User-Device; secure and no replay attacks
rule Send_UserDevice [color=#9AFF9A]:
    [Out_UD(<channelname,SendType,ReceiveType>,A,B,m)]
    -->
    [SecureUD(<channelname,SendType,ReceiveType>,A,B,m)]

rule Receive_UserDevice [color=#9AFF9A]:
    [SecureUD(<channelname,SendType,ReceiveType>,A,B,m)]
    -->
    [In_UD(<channelname,SendType,ReceiveType>,A,B,m)]


/*******************
*  Initialization  *
*******************/

divert(-1)
changequote(<!,!>)
define(Init_Provisioner,
<!rule Provisioner_Init [color=#FFEFD5]:
    let
        ReadStaticOOB = '$1'
        ReadOOBPK = '$2'
        SecureOption = '1'
    in
    [
        Fr(~NetKey)
    ]
    --[
        Atomic(~NetKey),
        OnlyOneProvisioner(),
        InitP_ReadStaticOOB('$1'),
        InitP_ReadOOBPK('$2')
    ]->
    [
        !Provisioner($P,~NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,'Provisioner')
    ]

restriction OnlyOneProvisioner:
    "All #i #j. OnlyOneProvisioner()@#i & OnlyOneProvisioner()@#j
        ==> #i = #j"!>)

define(Init_NewDevice,
<!rule NewDevice_Init_0 [color=#FFEFD5]:
    let
        /* PKType = 'InBand' */
        PKType = '$1'       /* 'InBand' or 'OOB' */
        OutputOOB = '$2'
        InputOOB = '$3'
        StaticOOB = '$4'
    in
    [
        Fr(~DeviceUUID)
    ]
    --[
        OnlyOneNewDevice0(),
        InitD_PKType('$1'),
        InitD_OutputOOB('$2'),
        InitD_InputOOB('$3'),
        InitD_StaticOOB('$4')
    ]->
    [
        NewDevice0($D,~DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,'NewDevice')
    ]

restriction OnlyOneNewDevice0:
    "All #i #j. OnlyOneNewDevice0()@#i & OnlyOneNewDevice0()@#j
        ==> #i = #j"!>)
changequote

divert(0)dnl
Init_Provisioner(ARGPStaticOOB,ARGPOOBPK)

Init_NewDevice(ARGDPKType,ARGDOutputOOB,ARGDInputOOB,ARGDStaticOOB)

rule NewDevice_Init_InBandPK_NoStaticOOB [color=#FFEFD5]:
    let
        skD = 'NULL'
        DHpkD = 'NULL'
        OOBPKURI = 'NULL'
        AuthValue = 'NULL'
        AuthValueURI = 'NULL'
    in
    [
        NewDevice0(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,'NewDevice')
    ]
    --[
        Eq(PKType,'InBand'),
        Eq(StaticOOB,'0'),
        OnlyOneNewDevice()
    ]->
    [
        !NewDevice(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,'NewDevice')
    ]

rule NewDevice_Init_OOBPK_NoStaticOOB [color=#FFEFD5]:
    let
        skD = ~skD
        DHpkD = 'g'^~skD
        OOBPKURI = ~URItoPK
        AuthValue = 'NULL'
        AuthValueURI = 'NULL'
    in
    [
        Fr(~skD),
        Fr(~URItoPK),
        NewDevice0(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,'NewDevice')
    ]
    --[
        Eq(PKType,'OOB'),
        Eq(StaticOOB,'0'),
        OnlyOneNewDevice()
    ]->
    [
        !NewDevice(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,'NewDevice'),
        !OOBPKI(NDev,DeviceUUID,DHpkD,OOBPKURI,'OOBPKI')
    ]

rule NewDevice_Init_InBandPK_StaticOOB [color=#FFEFD5]:
    let
        skD = 'NULL'
        DHpkD = 'NULL'
        OOBPKURI = 'NULL'
        AuthValue = ~random
        AuthValueURI = ~URItoAuth
    in
    [
        Fr(~random),
        Fr(~URItoAuth),
        NewDevice0(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,'NewDevice')
    ]
    --[
        Eq(PKType,'InBand'),
        Eq(StaticOOB,'1'),
        OnlyOneNewDevice()
    ]->
    [
        !NewDevice(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,'NewDevice'),
        !StaticAuthInfo(NDev,DeviceUUID,AuthValue,AuthValueURI,'StaticAuthInfo')
    ]

rule NewDevice_Init_OOBPK_StaticOOB [color=#FFEFD5]:
    let
        skD = ~skD
        DHpkD = 'g'^~skD
        OOBPKURI = ~URItoPK
        AuthValue = ~random
        AuthValueURI = ~URItoAuth
    in
    [
        Fr(~skD),
        Fr(~URItoPK),
        Fr(~random),
        Fr(~URItoAuth),
        NewDevice0(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,'NewDevice')
    ]
    --[
        Eq(PKType,'OOB'),
        Eq(StaticOOB,'1'),
        OnlyOneNewDevice()
    ]->
    [
        !NewDevice(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,'NewDevice'),
        !OOBPKI(NDev,DeviceUUID,DHpkD,OOBPKURI,'OOBPKI'),
        !StaticAuthInfo(NDev,DeviceUUID,AuthValue,AuthValueURI,'StaticAuthInfo')
    ]

restriction OnlyOneNewDevice:
    "All #i #j. OnlyOneNewDevice()@#i & OnlyOneNewDevice()@#j
        ==> #i = #j"

rule Init_User [color=#FFEFD5]:
    []
    --[ OnlyOneUser() ]->
    [ !User($User) ]

restriction OnlyOneUser:
    "All #i #j. OnlyOneUser()@#i & OnlyOneUser()@#j ==> #i = #j"


/**************
*  Beaconing  *
**************/

rule NDev_Beaconing [color=#FFF68F]:
    [
        !NewDevice(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,'NewDevice')
    ]
    --[
        OnlyOnceBeaconing()
    ]->
    [
        Out(<NDev,DeviceUUID,OOBPKURI,AuthValueURI>),
        NDevBeaconing(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI)
    ]

restriction OnlyOnceBeaconing:
    "All #i #j. OnlyOnceBeaconing()@#i & OnlyOnceBeaconing()@#j
        ==> #i = #j"


/***********************
*  Link Establishment  *
***********************/

rule Prov_LinkOpen [color=#BBFFFF]:
    let 
        LinkID = ~LinkID
    in
    [
        In(<NDev,DeviceUUID,OOBPKURI,AuthValueURI>),
        Fr(~LinkID),
        !Provisioner(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,'Provisioner')
    ]
    --[
        Neq(Prov,NDev),
        OnlyOnceLinkOpen()
    ]->
    [
        Out(<LinkID,'LinkOpen',DeviceUUID>),
        ProvLinkOpening(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI)
    ]
restriction OnlyOnceLinkOpen:
    "All #i #j. OnlyOnceLinkOpen()@#i & OnlyOnceLinkOpen()@#j
        ==> #i = #j"

rule NDev_LinkAck [color=#FFF68F]:
    [
        In(<LinkID,'LinkOpen',DeviceUUID_In>),
        NDevBeaconing(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI)
    ]
    --[
        Eq(DeviceUUID_In,DeviceUUID)
    ]->
    [
        Out(<LinkID,'Ack'>),
        NDevLinkOpened(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID)
    ]

// Collect Static OOB info
rule Prov_CollectOOBInfo [color=#BBFFFF]:
    [
        ProvLinkOpening(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI)
    ]
    --[
        Neq(AuthValueURI,'NULL'),
        Eq(ReadStaticOOB,'1')
    ]->
    [
        Out_StaticOOB(<'OOBReq','Prov','OOB'>,NDev,AuthValueURI),
        ProvCollectingStaticOOB(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI)
    ]
        
rule StaticOOB_SendOOBInfo [color=#00BFFF]:
    [
        In_StaticOOB(<'OOBReq','Prov','OOB'>,NDev_In,AuthValueURI_In),
        !StaticAuthInfo(NDev,DeviceUUID,AuthValue,AuthValueURI,'StaticAuthInfo')
    ]
    --[
        Eq(NDev,NDev_In),
        Eq(AuthValueURI,AuthValueURI_In)
    ]->
    [
        Out_StaticOOB(<'OOBRes','OOB','Prov'>,NDev,AuthValue)
    ]

rule Prov_StaticAuthValueCollected [color=#BBFFFF]:
    [
        In_StaticOOB(<'OOBRes','OOB','Prov'>,NDev,AuthValue),
        ProvCollectingStaticOOB(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI)
    ]
    -->
    [
        ProvLinkOpened(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]

// Cannot collect static oob info
rule Prov_LinkOpened [color=#BBFFFF]:
    let
        AuthValue = 'NULL'
    in
    [
        In(<LinkID,'Ack'>),
        ProvLinkOpening(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI)
    ]
    --[
        CannotCollect(ReadStaticOOB,AuthValueURI)
    ]->
    [
        !Provisioning(Prov,NDev),
        ProvLinkOpened(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]

restriction CannotCollect:
    "All r a #i. CannotCollect(r,a) @#i ==>
        (r = '0' | a = 'NULL')"


/***************
*  Invitation  *
***************/

rule Prov_Invite [color=#BBFFFF]:
    [
        ProvLinkOpened(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    -->
    [
        Out(<LinkID,'Invite'>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]

rule NDev_Capabilities [color=#FFF68F]:
    [
        In(<LinkID,'Invite'>),
        NDevLinkOpened(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID)
    ]
    -->
    [
        Out(<LinkID,PKType,OutputOOB,InputOOB,StaticOOB>),
        NDevSentCapabilities(NDev,DeviceUUID,PKType,OutputOOB,InputOOB,StaticOOB,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID)
    ]


/**********
*  Start  *
**********/

rule Prov_Chose_InBandPK_OutputOOB [color=#BBFFFF]:
    let
        PKType_P = 'InBand'
        AuthMethod_P = 'OutputOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        /* Neq(SecureOption, '1'), */
        UseInBandPK(PKType_D,ReadOOBPK),
        UseOutputOOB(OutputOOB_D,InputOOB_D,StaticOOB_D,AuthValue)
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule Prov_Chose_OOBPK_OutputOOB [color=#BBFFFF]:
    let
        PKType_P = 'OOB'
        AuthMethod_P = 'OutputOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        UseOOBPK(PKType_D,ReadOOBPK),
        UseOutputOOB(OutputOOB_D,InputOOB_D,StaticOOB_D,AuthValue)
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule Prov_Chose_InBandPK_InputOOB [color=#BBFFFF]:
    let
        PKType_P = 'InBand'
        AuthMethod_P = 'InputOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        /* Neq(SecureOption, '1'), */
        UseInBandPK(PKType_D,ReadOOBPK),
        UseInputOOB(OutputOOB_D,InputOOB_D,StaticOOB_D,AuthValue)
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule Prov_Chose_OOBPK_InputOOB [color=#BBFFFF]:
    let
        PKType_P = 'OOB'
        AuthMethod_P = 'InputOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        UseOOBPK(PKType_D,ReadOOBPK),
        UseInputOOB(OutputOOB_D,InputOOB_D,StaticOOB_D,AuthValue)
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule Prov_Chose_InBandPK_StaticOOB [color=#BBFFFF]:
    let
        PKType_P = 'InBand'
        AuthMethod_P = 'StaticOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        Neq(SecureOption, '1'),
        UseInBandPK(PKType_D,ReadOOBPK),
        Eq(StaticOOB_D, '1'),
        Neq(AuthValue, 'NULL')
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule Prov_Chose_OOBPK_StaticOOB [color=#BBFFFF]:
    let
        PKType_P = 'OOB'
        AuthMethod_P = 'StaticOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        UseOOBPK(PKType_D,ReadOOBPK),
        Eq(StaticOOB_D, '1'),
        Neq(AuthValue, 'NULL')
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule Prov_Chose_InBandPK_NoOOB [color=#BBFFFF]:
    let
        PKType_P = 'InBand'
        AuthMethod_P = 'NoOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        Neq(SecureOption, '1'),
        UseInBandPK(PKType_D,ReadOOBPK),
        UseNoOOB(OutputOOB_D,InputOOB_D,StaticOOB_D,AuthValue)
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule Prov_Chose_OOBPK_NoOOB [color=#BBFFFF]:
    let
        PKType_P = 'OOB'
        AuthMethod_P = 'NoOOB'
    in
    [
        In(<LinkID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D>),
        ProvInvited(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue)
    ]
    --[
        Neq(SecureOption, '1'),
        UseOOBPK(PKType_D,ReadOOBPK),
        UseNoOOB(OutputOOB_D,InputOOB_D,StaticOOB_D,AuthValue)
    ]->
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

restriction UseInBandPK:
    "All p r #i. UseInBandPK(p,r)@#i ==>
        (p = 'InBand' | r = '0')"
restriction UseOOBPK:
    "All p r #i. UseOOBPK(p,r)@#i ==>
        (p = 'OOB' & r = '1')"
restriction UseOutputOOB:
    "All o i s a #j. UseOutputOOB(o,i,s,a)@#j ==>
        (s = '0' | a = 'NULL') & o = '1'"
restriction UseInputOOB:
    "All o i s a #j. UseInputOOB(o,i,s,a)@#j ==>
        (s = '0' | a = 'NULL') & o = '0' & i = '1'"
restriction UseNoOOB:
    "All o i s a #j. UseNoOOB(o,i,s,a)@#j ==>
        (s = '0' | a = 'NULL') & o = '0' & i = '0'"

rule Prov_SendStart [color=#BBFFFF]:
    [
        ProvChosed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]
    --[
    ]->
    [
        Out(<LinkID,PKType_P,AuthMethod_P>),
        ProvSentStart(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule NDev_ReceiveStart [color=#FFF68F]:
    [
        In(<LinkID,PKType_P,AuthMethod_P>),
        NDevSentCapabilities(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID)
    ]
    --[
        ValidityCheck(PKType_P,AuthMethod_P,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D)
    ]->
    [
        NDevReceivedStart(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P)
    ]

restriction ValidityCheck:
    "All ppk pam dpk dout din dst #j. ValidityCheck(ppk,pam,dpk,dout,din,dst)@#j ==>
        (
            (ppk = 'InBand' | dpk = 'OOB') &
            ((pam = 'OutputOOB' & dout = '1') |
             (pam = 'InputOOB' & din = '1')  |
             (pam = 'StaticOOB' & dst = '1')  |
             (pam = 'NoOOB'))
        )"

/************************
*  Public Key Exchange  *
************************/
// InBand PK-EX
rule Prov_SendPK_InBand [color=#BBFFFF]:
    let
        skP = ~sk
        DHpkP = 'g'^~sk
    in
    [
        Fr(~sk),
        ProvSentStart(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]
    --[
        Eq(PKType_P,'InBand'),
        ProvInBandPK()
    ]->
    [
        Out(<LinkID,DHpkP>),
        ProvSentPKInBand(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP)
    ]

rule NDev_SendPK_InBand [color=#FFF68F]:
    let
        skD = ~sk
        DHpkD = 'g'^~sk
        OOBPKURI = OOBPKURI_NULL
    in
    [
        In(<LinkID,DHpkP>),
        Fr(~sk),
        NDevReceivedStart(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD_NULL,DHpkD_NULL,OOBPKURI_NULL,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P)
    ]
    --[
        Eq(PKType_P,'InBand'),
        NDevInBandPK()
    ]->
    [
        Out(<LinkID,DHpkD>),
        NDevExchanged(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP)
    ]

rule Prov_ReceivePK_InBand [color=#BBFFFF]:
    [
        In(<LinkID,DHpkD>),
        ProvSentPKInBand(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP)
    ]
    -->
    [
        ProvExchanged(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD)
    ]

// OOBPK-EX
rule Prov_ReqPK_OOB [color=#BBFFFF]:
    [
        ProvSentStart(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]
    --[
        Eq(PKType_P,'OOB'),
        Neq(OOBPKURI,'NULL'),
        ProvOOBPK()
    ]->
    [
        Out_OOBPK(<'OOBReq','Prov','OOB'>,NDev,OOBPKURI),
        ProvReqOOBPK(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]

rule OOBPKI_SendPK [color=#00BFFF]:
    [
        In_OOBPK(<'OOBReq','Prov','OOB'>,NDev_In,OOBPKURI_In),
        !OOBPKI(NDev,DeviceUUID,DHpkD,OOBPKURI,'OOBPKI')
    ]
    --[
        Eq(NDev,NDev_In),
        Eq(OOBPKURI,OOBPKURI_In)
    ]->
    [
        Out_OOBPK(<'OOBRes','OOB','Prov'>,NDev,DHpkD)
    ]

rule Prov_SendPK_OOB [color=#BBFFFF]:
    let
        skP = ~sk
        DHpkP = 'g'^~sk
    in
    [
        In_OOBPK(<'OOBRes','OOB','Prov'>,NDev,DHpkD),
        Fr(~sk),
        ProvReqOOBPK(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P)
    ]
    --[
    ]->
    [
        Out(<LinkID,DHpkP>),
        ProvExchanged(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD)
    ]

rule NDev_ReceivePK_OOB [color=#FFF68F]:
    [
        In(<LinkID,DHpkP>),
        NDevReceivedStart(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P)
    ]
    --[
        Eq(PKType_P,'OOB'),
        NDevOOBPK()
    ]->
    [
        NDevExchanged(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP)
    ]

/* ECDHSecret Calculation */
rule Prov_CalculateECDH [color=#BBFFFF]:
    let
        ECDHSecret = DHpkD^skP
    in
    [
        ProvExchanged(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD)
    ]
    --[
        Neq(DHpkP,DHpkD)
    ]->
    [
        ProvCalculatedECDH(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]

rule NDev_CalculateECDH [color=#FFF68F]:
    let
        ECDHSecret = DHpkP^skD
    in
    [
        NDevExchanged(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP)
    ]
    --[
        Neq(DHpkP,DHpkD)
    ]->
    [
        NDevCalculatedECDH(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]


/*******************
*  Authentication  *
*******************/
/* Output OOB */
rule NDev_DisplayAuthValue_OutputOOB [color=#FFF68F]:
    let
        AuthValue = ~random
        AuthValueURI = AuthValueURI_NULL
    in
    [
        Fr(~random),
        NDevCalculatedECDH(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue_NULL,AuthValueURI_NULL,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]
    --[
        Eq(AuthMethod_P,'OutputOOB'),
        NDevOutputOOB(),
        NDEV_DIS_AUTHVALUE(AuthValue)
    ]->
    [
        Out_UD(<'Display','Device','User'>,NDev,$User,~random),
        NDevWaitConfirm(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]

rule Prov_AskForInput_OutputOOB [color=#BBFFFF]:
    [
        ProvCalculatedECDH(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]
    --[
        Eq(AuthMethod_P,'OutputOOB'),
        ProvOutputOOB()
    ]->
    [
        Out_UD(<'AskForInput','Device','User'>,Prov,$User,'Input'),
        ProvAskedForInput(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]

rule Prov_GetAuthValue_OutputOOB [color=#BBFFFF]:
    let
        AuthValue = random
        AuthValueURI = AuthValueURI_NULL
    in
    [
        In_UD(<'Input','User','Device'>,$User,Prov,random),
        ProvAskedForInput(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI_NULL,AuthValue_NULL,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]
    --[
    ]->
    [
        ProvReadyForConfirm(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]

/* Input OOB */
rule Prov_DisplayAuthValue_InputOOB [color=#BBFFFF]:
    let
        AuthValue = ~random
        AuthValueURI = AuthValueURI_NULL
    in
    [
        Fr(~random),
        ProvCalculatedECDH(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI_NULL,AuthValue_NULL,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]
    --[
        Eq(AuthMethod_P,'InputOOB'),
        ProvInputOOB(),
        PROV_DIS_AUTHVALUE(AuthValue)
    ]->
    [
        Out_UD(<'Display','Device','User'>,Prov,$User,~random),
        ProvWaitInputComplete(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]

rule NDev_AskForInput_InputOOB [color=#FFF68F]:
    [
        NDevCalculatedECDH(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]
    --[
        Eq(AuthMethod_P,'InputOOB'),
        NDevInputOOB()
    ]->
    [
        Out_UD(<'AskForInput','Device','User'>,NDev,$User,'Input'),
        NDevAskForInput(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]

rule NDev_GetAuthValue_InputOOB [color=#FFF68F]:
    let
        AuthValue = random
        AuthValueURI = AuthValueURI_NULL
    in
    [
        In_UD(<'Input','User','Device'>,$User,NDev,random),
        NDevAskForInput(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue_NULL,AuthValueURI_NULL,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]
    --[
        NDEV_ASKFORINPUT(NDev,LinkID)
    ]->
    [
        Out(<LinkID,'InputComplete'>),
        NDevWaitConfirm(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]

rule Prov_ReadyForConfirm_InputOOB [color=#BBFFFF]:
    [
        In(<LinkID,'InputComplete'>),
        ProvWaitInputComplete(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]
    -->
    [
        ProvReadyForConfirm(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]

// Static OOB
rule Prov_GetAuthValue_StaticOOB [color=#BBFFFF]:
    [   
        ProvCalculatedECDH(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]
    --[
        PROV_STA_AUTHVALUE(AuthValue),
        Eq(AuthMethod_P,'StaticOOB'),
        ProvStaticOOB()
    ]->
    [
        ProvReadyForConfirm(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]

rule NDev_GetAuthValue_StaticOOB [color=#FFF68F]:
    [
        NDevCalculatedECDH(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]
    --[
        NDEV_STA_AUTHVALUE(AuthValue),
        Eq(AuthMethod_P,'StaticOOB'),
        NDevStaticOOB()
    ]->
    [
        NDevWaitConfirm(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]
    
// No OOB
rule Prov_GetAuthValue_NoOOB [color=#BBFFFF]:
    let
        AuthValue = '000000'
        AuthValueURI = AuthValueURI_NULL
    in
    [   
        ProvCalculatedECDH(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI_NULL,AuthValue_NULL,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]
    --[
        Eq(AuthMethod_P,'NoOOB'),
        ProvNoOOB(),
        PROV_NOO_AUTHVALUE(AuthValue)
    ]->
    [
        ProvReadyForConfirm(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]

rule NDev_GetAuthValue_NoOOB [color=#FFF68F]:
    let
        AuthValue = '000000'
        AuthValueURI = AuthValueURI_NULL
    in
    [
        NDevCalculatedECDH(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue_NULL,AuthValueURI_NULL,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]
    --[
        Eq(AuthMethod_P,'NoOOB'),
        NDevNoOOB(),
        NDEV_NOO_AUTHVALUE(AuthValue)
    ]->
    [
        NDevWaitConfirm(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]

/* User Interaction */
rule User_OutputOOB [color=#6495ED]:
    [
        !User(User),
        !Provisioning(D1,D2),
        In_UD(<'AskForInput','Device','User'>,D1,User,'Input'),
        In_UD(<'Display','Device','User'>,D2,User,m)
    ]
    --[
        OneInteraction()
    ]->
    [
        Out_UD(<'Input','User','Device'>,User,D1,m)
    ]
rule User_InputOOB [color=#6495ED]:
    [
        !User(User),
        !Provisioning(D1,D2),
        In_UD(<'Display','Device','User'>,D1,User,m),
        In_UD(<'AskForInput','Device','User'>,D2,User,'Input')
    ]
    --[
        USER_INPUT(D2),
        OneInteraction()
    ]->
    [
        Out_UD(<'Input','User','Device'>,User,D2,m)
    ]
rule User_Abnormal [color=#6495ED]:
    [
        !User(User),
        !Provisioning(D1,D2),
        Fr(~m),
        In_UD(<'AskForInput','Device','User'>,D1,User,'Input'),
        In_UD(<'AskForInput','Device','User'>,D2,User,'Input')
    ]
    --[
        USER_ABNORM(~m),
        OneInteraction()
    ]->
    [
        Out_UD(<'Input','User','Device'>,User,D1,~m),
        Out_UD(<'Input','User','Device'>,User,D2,~m)
    ]
restriction OneInteraction:
    "All #i #j. OneInteraction()@#i & OneInteraction()@#j ==> #i = #j"


/* Same below */
rule Prov_Confirmation [color=#BBFFFF]:
    let
        Np = <~Np,'1'>
        AuthValue = <AuthValue_In,'2'>
        ConfirmSALT = s1(<DHpkP,DHpkD>)
        ECDH1 = <fst(ECDHSecret),'1'>
        ECDH2 = <snd(ECDHSecret),'2'>
        Tk = cmac(ECDH1+ECDH2,ConfirmSALT)
        CK = cmac('prck',Tk)
        Cp = cmac(Np+AuthValue, CK)
    in
    [
        Fr(~Np),
        ProvReadyForConfirm(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue_In,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret)
    ]
    --[
        PROV_GET_AUTHVALUE(AuthValue_In),
        Running_Prov(Prov,NDev,<ECDHSecret,Np>)
    ]->
    [
        Out(<LinkID,Cp>),
        ProvSentConfirmation(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp)
    ]

rule NDev_Confirmation [color=#FFF68F]:
    let
        Nd = <~Nd,'1'>
        AuthValue = <AuthValue_In,'2'>
        ConfirmSALT = s1(<DHpkP,DHpkD>)
        ECDH1 = <fst(ECDHSecret),'1'>
        ECDH2 = <snd(ECDHSecret),'2'>
        Tk = cmac(ECDH1+ECDH2,ConfirmSALT)
        CK = cmac('prck',Tk)
        Cd = cmac(Nd+AuthValue, CK)
    in
    [
        In(<LinkID,Cp>),
        Fr(~Nd),
        NDevWaitConfirm(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue_In,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret)
    ]
    --[
        Neq(Cp,Cd),
        NDEV_GET_AUTHVALUE(AuthValue_In),
        Running_NDev(NDev,<ECDHSecret,Nd>)
    ]->
    [
        Out(<LinkID,Cd>),
        NDevSentConfirm(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret,Cp,Nd,ConfirmSALT,ECDH1,ECDH2,CK,Cd)
    ]

rule Prov_SendNonce [color=#BBFFFF]:
    [
        In(<LinkID,Cd>),
        ProvSentConfirmation(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp)
    ]
    --[
        Neq(Cp,Cd)
    ]->
    [
        Out(<LinkID,Np>),
        ProvSentNonce(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp,Cd)
    ]

rule NDev_SendNonce [color=#FFF68F]:
    let
        ProvisioningSALT = s1(<ConfirmSALT,Np,Nd>)
        /* SessionKey = k1(ECDHSecret,ProvisioningSALT,'prsk') */
        Tk = cmac(ECDH1+ECDH2,ProvisioningSALT)
        SessionKey = cmac('prsk',Tk)
    in
    [
        In(<LinkID,Np>),
        NDevSentConfirm(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret,Cp,Nd,ConfirmSALT,ECDH1,ECDH2,CK,Cd)
    ]
    --[
        Check(Cp, Np+AuthValue, CK)
    ]->
    [
        Out(<LinkID,Nd>),
        NDevSentNonce(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret,Cp,Nd,ConfirmSALT,ECDH1,ECDH2,CK,Cd,Np,ProvisioningSALT,SessionKey)
    ]

rule Prov_Check [color=#BBFFFF]:
    let
        ProvisioningSALT = s1(<ConfirmSALT,Np,Nd>)
        /* SessionKey = k1(ECDHSecret,ProvisioningSALT,'prsk') */
        Tk = cmac(ECDH1+ECDH2,ProvisioningSALT)
        SessionKey = cmac('prsk',Tk)
    in
    [
        In(<LinkID,Nd>),
        ProvSentNonce(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp,Cd)
    ]
    --[
        Check(Cd, Nd+AuthValue, CK)
    ]->
    [
        ProvChecked(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp,Cd,Nd,ProvisioningSALT,SessionKey)
    ]

restriction CheckCMAC:
    "All C M K #i. Check(C,M,K) @#i
        ==> 
        ( C = cmac(M,K) )
        | (Ex mk mo. ( mk + mo = M 
                    & ( not (Ex x y. x + y = mk ))
                    & ( mk = rm(C,mo,K) )
                    )
          )
        | ( (not (Ex x y. x + y = M))
            & (M = rm1(C,K))
          )
    "

/*****************
*  Distribution  *
*****************/
rule Prov_Distribute [color=#BBFFFF]:
    let
        ProvData = senc(<NetKey>,SessionKey)
        /* DevKey = k1(ECDHSecret,ProvisioningSALT,'prdk') */
        Tk = cmac(ECDH1+ECDH2,ProvisioningSALT)
        DevKey = cmac('prdk',Tk)
    in
    [
        ProvChecked(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp,Cd,Nd,ProvisioningSALT,SessionKey)
    ]
    --[
        Secret(SessionKey,DevKey,NetKey),
        Role('Prov')
    ]->
    [
        Out(<LinkID,ProvData>),
        ProvDistributed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp,Cd,Nd,ProvisioningSALT,SessionKey,ProvData,DevKey)
    ]

rule NDev_Complete [color=#FFF68F]:
    let
        ProvData = senc(<NetKey>,SessionKey)
        /* DevKey = k1(ECDHSecret,ProvisioningSALT,'prdk') */
        Tk = cmac(ECDH1+ECDH2,ProvisioningSALT)
        DevKey = cmac('prdk',Tk)
    in
    [
        In(<LinkID,ProvData>),
        NDevSentNonce(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret,Cp,Nd,ConfirmSALT,ECDH1,ECDH2,CK,Cd,Np,ProvisioningSALT,SessionKey)
    ]
    --[
        FinishedD(),
        Commit_NDev(NDev,<ECDHSecret,Np>),
        Role('NDev')
    ]->
    [
        Out(<LinkID,'Complete'>),
        NDevComplete(NDev,DeviceUUID,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,skD,DHpkD,OOBPKURI,AuthValue,AuthValueURI,LinkID,PKType_P,AuthMethod_P,DHpkP,ECDHSecret,Cp,Nd,ConfirmSALT,ECDH1,ECDH2,CK,Cd,Np,ProvisioningSALT,SessionKey,DevKey,NetKey)
    ]

rule Prov_Complete [color=#BBFFFF]:
    [
        In(<LinkID, 'Complete'>),
        ProvDistributed(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp,Cd,Nd,ProvisioningSALT,SessionKey,ProvData,DevKey)
    ]
    --[
        FinishedP(),
        Commit_Prov(Prov,NDev,<ECDHSecret,Nd>)
    ]->
    [
        ProvComplete(Prov,NetKey,ReadOOBPK,ReadStaticOOB,SecureOption,NDev,DeviceUUID,LinkID,OOBPKURI,AuthValueURI,AuthValue,PKType_D,OutputOOB_D,InputOOB_D,StaticOOB_D,PKType_P,AuthMethod_P,skP,DHpkP,DHpkD,ECDHSecret,Np,ConfirmSALT,ECDH1,ECDH2,CK,Cp,Cd,Nd,ProvisioningSALT,SessionKey,ProvData,DevKey)
    ]



restriction Equality:
    "All x y #i. Eq(x,y) @#i ==> x = y"
restriction Inequality:
    "All x #i. Neq(x,x) @#i ==> F"

lemma types [sources]:
    " 
    (All r1 #i. PROV_GET_AUTHVALUE(r1) @i
        ==>
        ( (Ex #j. PROV_DIS_AUTHVALUE(r1) @j)
        | (Ex #j. NDEV_DIS_AUTHVALUE(r1) @j)
        | (Ex #j. PROV_NOO_AUTHVALUE(r1) @j)
        | (Ex #j. PROV_STA_AUTHVALUE(r1) @j)
        | (Ex #j. USER_ABNORM(r1) @j)
        )
    )
    & (All r2 #i. NDEV_GET_AUTHVALUE(r2) @i
        ==>
        ( (Ex #j. PROV_DIS_AUTHVALUE(r2) @j)
        | (Ex #j. NDEV_DIS_AUTHVALUE(r2) @j)
        | (Ex #j. NDEV_NOO_AUTHVALUE(r2) @j)
        | (Ex #j. NDEV_STA_AUTHVALUE(r2) @j)
        | (Ex #j. USER_ABNORM(r2) @j)
        )
      )
    & 
    (All ndev linkid #i. NDEV_ASKFORINPUT(ndev,linkid) @i
        ==>
        ( ( (Ex #j. USER_INPUT(ndev) @j) | (Ex m #l. USER_ABNORM(m) @l) )
        & (Ex #k. KU(linkid) @k & k < i)
        )
    )
    "

lemma executability:
    exists-trace
    "Ex #i #j. FinishedP() @i & FinishedD()@j
    &
    (All #m. InitP_ReadOOBPK('0') @m ==> (Ex #p #q. (ProvInBandPK() @p & NDevInBandPK() @q)))
    &
    (All #m #n. InitP_ReadOOBPK('1') @m & InitD_PKType('InBand') @n ==> (Ex #p #q. (ProvInBandPK() @p & NDevInBandPK() @q)))
    &
    (All #m #n. InitP_ReadOOBPK('1') @m & InitD_PKType('OOB') @n ==> (Ex #p #q. (ProvOOBPK() @p & NDevOOBPK() @q)))
    &
    (All #m #n. InitP_ReadStaticOOB('0') @m & InitD_OutputOOB('1') @n ==> (Ex #p #q. (ProvOutputOOB() @p & NDevOutputOOB() @q)))
    &
    (All #l #m #n. InitP_ReadStaticOOB('1') @l & InitD_OutputOOB('1') @m & InitD_StaticOOB('0') @n ==> (Ex #p #q. (ProvOutputOOB() @p & NDevOutputOOB() @q)))
    &
    (All #l #m #n. InitP_ReadStaticOOB('0') @l & InitD_OutputOOB('0') @m & InitD_InputOOB('1') @n ==> (Ex #p #q. (ProvInputOOB() @p & NDevInputOOB() @q)))
    &
    (All #k #l #m #n. InitP_ReadStaticOOB('1') @k & InitD_OutputOOB('0') @l & InitD_InputOOB('1') @m & InitD_StaticOOB('0') @n ==> (Ex #p #q. (ProvInputOOB() @p & NDevInputOOB() @q)))
    &
    (All #m #n. InitP_ReadStaticOOB('1') @m & InitD_StaticOOB('1') @n ==> (Ex #p #q. (ProvStaticOOB() @p & NDevStaticOOB() @q)))
    &
    (All #l #m #n. InitP_ReadStaticOOB('0') @l & InitD_OutputOOB('0') @m & InitD_InputOOB('0') @n ==> (Ex #p #q. (ProvNoOOB() @p & NDevNoOOB() @q)))
    &
    (All #k #l #m #n. InitP_ReadStaticOOB('1') @k & InitD_OutputOOB('0') @l & InitD_InputOOB('0') @m & InitD_StaticOOB('0') @n ==> (Ex #p #q. (ProvNoOOB() @p & NDevNoOOB() @q)))
    "

/* lemma Noninj_Agreement_Np: */
/*     "All ndev np #i. */
/*         RecvNonce_NDev(ndev,np) @i */
/*         ==> (Ex prov #j. SendNonce_Prov(prov,ndev,np) @j) */
/*     " */

/* lemma Noninj_Agreement_Nd: */
/*     "All prov ndev nd #i. */
/*         RecvNonce_Prov(prov,ndev,nd) @i */
/*         ==> (Ex #j. SentNonce_NDev(ndev,nd) @j) */
/*     " */

lemma Noninj_Agreement_NDev:
    "All ndev m #i.
        Commit_NDev(ndev, m) @i
        ==> (Ex prov #j. Running_Prov(prov, ndev, m) @j)
    "

lemma Noninj_Agreement_Prov:
    "All prov ndev m #i.
        Commit_Prov(prov,ndev,m) @i
        ==> (Ex #j. Running_NDev(ndev, m) @j)
    "

lemma Secrecy_Keys:
    "All sessionkey devkey netkey #i.
        Secret(sessionkey,devkey,netkey) @i
        ==> (not ( (Ex #j. K(netkey)@j)
                 | (Ex #k. K(sessionkey)@k)
                 | (Ex #l. K(devkey)@l)
                 )
            )
    "

end
