import sys
import os
import numpy as np
import getopt
import json
from tqdm import tqdm



# python clustergenerator.py --maxclustersize=4 --molecules=HF.sdf
# python clustergenerator.py --gen3Dimages 
maxclustersize=3
molecules=None
molsperrow=3
outputfileformat='mol'
pairwisedistanceratio=1 # angstroms
outputfilepath='ClusteredStructures/'
gen3Dimages=False
donormaxedgesout=2
stericcutoff=1
maxgraphinvorder=2
opts, xargs = getopt.getopt(sys.argv[1:],'cs:mols',["maxclustersize=","molecules=","molsperrow=","outputfileformat=","pairwisedistanceratio=","outputfilepath=","gen3Dimages","stericcutoff=","maxgraphinvorder="])

for o, a in opts:
    if o in ("-mcs", "--maxclustersize"):
        maxclustersize=int(a)
    elif o in ("-mols","--molecules"):
        temp=[]
        molecules=a.split(',')
        for mol in molecules:
            newmol=mol.rstrip().lstrip()
            temp.append(newmol)
        molecules=temp
    elif o in ("--molsperrow"):
        molsperrow=int(a)
    elif o in ("--outputfileformat"):
        outputfileformat=a
    elif o in ("--pairwisedistance"):
        pairwisedistance=float(a)
    elif o in ("--outputfilepath"):
        outputfilepath=a
    elif o in ("--gen3Dimages"):
        gen3Dimages=True
    elif o in ("--stericcutoff"):
        stericcutoff=float(a)
    elif o in ("--maxgraphinvorder"):
        maxgraphinvorder=int(a)

def ReduceAtomMatrix(mat,moleculetoindexlist,moleculesgrown,indextomolecule):
    atomindextomoleculeindex={}
    molindextomoleculename={}
    molidx=0
    for molecule,indexlist in moleculetoindexlist.items():
        if molecule not in moleculesgrown:
            continue
        for index in indexlist:
            atomindextomoleculeindex[index]=molidx
        molindextomoleculename[molidx]=molecule.split('.')[0]
        molidx+=1
    pairwiselocs=[]
    locs=np.transpose(np.where(mat==1))
    for loc in locs:
        i=loc[0]
        j=loc[1]
        if mat[j,i]==1: # then bond and dont append
            continue
        else:
            pairwiselocs.append(loc)
    moleculenum=len(moleculesgrown)
    reducemat=np.zeros((moleculenum,moleculenum))
    for pair in pairwiselocs:
        i=pair[0]
        j=pair[1]
        firstmolecule=indextomolecule[i]
        secondmolecule=indextomolecule[j]
        rowindex=atomindextomoleculeindex[i]
        colindex=atomindextomoleculeindex[j]
        reducemat[rowindex,colindex]=1   
    return reducemat,molindextomoleculename 



def GenerateDictionaries(molecules,donormaxedgesout):
    moleculenametodics={}
    labeltodonoracceptorlabel={'H':'D','F':'A','O':'A','S':'A','N':'A','C':'D'}
    labeltovalenceelectrons={'H':1,'C':4,'O':6,'S':6,'F':7}
    atomicsymboltovdwradius= {"H" : 1.20, "Li": 1.82, "Na": 2.27, "K": 2.75, "Rb": 3.03, "Cs": 3.43,"Be": 1.53, "Mg": 1.73, "Ca": 2.31, "B": 1.92, "C": 1.70, "N": 1.55, "O":1,"P" : 1.80, "S" : 1.80, "F" : 1.47, "Cl":1.75, "Br":1.85, "Zn":1.39}
    atomicsymboltoelectronegativity={"H":2.2,"C":2.55,"O":3.44,"S":2.58,"F":3.98} 
    moleculetoatomindextomaxedgesin={}
    moleculetoatomindextomaxedgesout={}
    moleculetoatomindextolabelgraphs={}
    moleculetoatomindextolabelrdkit={}
    moleculetonumberofatoms={}
    moleculetoatomindextosymclass={}
    moleculetoatomindextoatomicsymbol={}
    moleculetobondconnectivitylist={} # need this to post process add arrows to digraph matrix (cant have row of zeros so need arrows even for inside molecules, not just between molecules
    moleculetoOBmol={}
    moleculetoatomicpairtodistance={}
    moleculetoatomindextoinitialposition={}
    symclassesused=[]
    moleculenames=[]
    for molecule in molecules:
        atomindextolabelgraphs={}
        atomindextolabelrdkit={}
        atomindextoatomicsymbol={}
        atomindextomaxedgesin={}
        atomindextomaxedgesout={}
        atomicpairtodistance={}
        atomindextomaxedgesin={}
        atomindextomaxedgesout={}
        atomicpairtodistance={}
        atomindextoinitialposition={} # when updating mol properties and a pairwise molecule repeats, need to set back to original coordinates
        obConversion = openbabel.OBConversion()
        mol = openbabel.OBMol()
        inFormat = obConversion.FormatFromExt(molecule)
        obConversion.SetInFormat(inFormat)
        obConversion.ReadFile(mol,molecule)
        moleculetoOBmol[molecule]=mol
        atomiter=openbabel.OBMolAtomIter(mol)
        etab = openbabel.OBElementTable()
        numberofatoms=mol.NumAtoms()
        idxtosymclass,symclassesused=gen_canonicallabels(mol,symclassesused)
        bondconnectivitylist=[]
        bonditer=openbabel.OBMolBondIter(mol)
        for bond in bonditer:
            bgnatomidx=bond.GetBeginAtomIdx()-1
            endatomidx=bond.GetEndAtomIdx()-1
            fwd=[bgnatomidx,endatomidx]
            rev=[endatomidx,bgnatomidx]
            bondconnectivitylist.append(fwd)
            bondconnectivitylist.append(rev)
        for atom in atomiter:
            atomidx=atom.GetIdx()
            initialposition=[atom.GetX(),atom.GetY(),atom.GetZ()]
            atomindextoinitialposition[atomidx]=initialposition
            atomlabel=etab.GetSymbol(atom.GetAtomicNum())
            symclass=idxtosymclass[atomidx-1]
            atomindextolabelgraphs[atomidx-1]=atomlabel
            atomindextolabelrdkit[atomidx-1]=str(symclass)

            atomindextoatomicsymbol[atomidx-1]=atomlabel
            atomatomiter=openbabel.OBAtomAtomIter(atom)
            bondedelectrons=0
            donoracceptorlabel=labeltodonoracceptorlabel[atomlabel]
            if donoracceptorlabel=='A':
                for natom in atomatomiter:
                    natomidx=natom.GetIdx()
                    bond=mol.GetBond(atomidx,natomidx)
                    bondorder=bond.GetBondOrder()
                    bondedelectrons+=bondorder
                valenceelectrons=labeltovalenceelectrons[atomlabel]
                remainder=valenceelectrons-bondedelectrons
                lonepairs=remainder/2
                maxedgesin=lonepairs
                maxedgesout=0
            else:
                maxedgesin=0
                maxedgesout=donormaxedgesout
            atomindextomaxedgesin[atomidx-1]=maxedgesin
            atomindextomaxedgesout[atomidx-1]=maxedgesout
            anotheratomiter=openbabel.OBMolAtomIter(mol)
            for anotheratom in anotheratomiter:
                anotheratomidx=anotheratom.GetIdx()
                if anotheratomidx!=atomidx:
                    atomvec=np.array([atom.GetX(),atom.GetY(),atom.GetZ()])
                    anotheratomvec=np.array([anotheratom.GetX(),anotheratom.GetY(),anotheratom.GetZ()])
                    dist=np.linalg.norm(atomvec-anotheratomvec)
                    atomicpair=tuple([atomidx,anotheratomidx])
                    if atomicpair not in atomicpairtodistance.keys():
                        atomicpairtodistance[atomicpair]=dist

        moleculetoatomindextomaxedgesin[molecule]=atomindextomaxedgesin
        moleculetoatomindextomaxedgesout[molecule]=atomindextomaxedgesout
        moleculetoatomindextolabelgraphs[molecule]=atomindextolabelgraphs
        moleculetoatomindextolabelrdkit[molecule]=atomindextolabelrdkit

        moleculetonumberofatoms[molecule]=numberofatoms
        moleculetoatomindextosymclass[molecule]=idxtosymclass
        moleculetobondconnectivitylist[molecule]=bondconnectivitylist
        moleculenames.append(molecule)
        moleculetoatomindextoatomicsymbol[molecule]=atomindextoatomicsymbol
        moleculetoatomicpairtodistance[molecule]=atomicpairtodistance
        moleculetoatomindextoinitialposition[molecule]=atomindextoinitialposition
    moleculenametodics['atomindextomaxedgesin']=moleculetoatomindextomaxedgesin
    moleculenametodics['atomindextomaxedgesout']=moleculetoatomindextomaxedgesout
    moleculenametodics['atomindextolabelgraphs']=moleculetoatomindextolabelgraphs
    moleculenametodics['atomindextolabelrdkit']=moleculetoatomindextolabelrdkit
    moleculenametodics['numberofatoms']=moleculetonumberofatoms
    moleculenametodics['atomindextosymclass']=moleculetoatomindextosymclass
    moleculenametodics['moleculenames']=moleculenames
    moleculenametodics['bondconnectivitylist']=moleculetobondconnectivitylist
    moleculenametodics['atomindextoatomicsymbol']=moleculetoatomindextoatomicsymbol
    moleculenametodics['OBmol']=moleculetoOBmol
    moleculenametodics['atomicpairtodistance']=moleculetoatomicpairtodistance
    moleculenametodics['labeltodonoracceptorlabel']=labeltodonoracceptorlabel
    moleculenametodics['atomicsymboltovdwradius']=atomicsymboltovdwradius
    moleculenametodics['atomindextoinitialposition']=moleculetoatomindextoinitialposition
    moleculenametodics['atomicsymboltoelectronegativity']=atomicsymboltoelectronegativity
    return moleculenametodics
 
def AdjacencyMatrix(bondarray,locsintermolecularlist,atomsum):
    array=np.zeros((atomsum,atomsum))
    index=0
    for j in range(len(array)-1):
        startindex=j+1
        for k in range(startindex,len(array[0])):
            if [j,k] in locsintermolecularlist: # then add back to matrix
                value=int(bondarray[index])
                array[j,k]=value
                index+=1
                
    return array


def DuplicateMoleculesAndMolecularPropertyDictionaries(moleculenames,n,moleculetoatomindextomaxedgesin,moleculetoatomindextomaxedgesout,moleculetoatomindextolabelgraphs,moleculetoatomindextolabelrdkit,moleculetonumberofatoms,moleculetoatomindextosymclass,moleculetobondconnectivitylist,moleculetoatomindextoatomicsymbol,moleculetoOBmol,moleculetoatomicpairtodistance,moleculetoatomindextoinitialposition):
    # for duplicates of molecule, will have to make new molecule name but copy all dictionary properties in order for combinations function from itertools to work
    moleculessametype=[]
    moleculelist=[]
    for molecule in moleculenames:
        moleculelist.append(molecule)
    tempmoleculelist=[]
    for molecule in moleculelist:
        tempmoleculelist.append(molecule)
        if n>1: 
            temp=[molecule]
            for i in range(n-1): # if n=2 that would be too much for double since original is already a copy
                moleculesplit=molecule.split('.')
                newmolecule=moleculesplit[0]+'_'+str(i)+'.'+moleculesplit[1]
                moleculetoatomindextomaxedgesin[newmolecule]=moleculetoatomindextomaxedgesin[molecule]
                moleculetoatomindextomaxedgesout[newmolecule]=moleculetoatomindextomaxedgesout[molecule]
                moleculetoatomindextolabelgraphs[newmolecule]=moleculetoatomindextolabelgraphs[molecule]
                moleculetoatomindextolabelrdkit[newmolecule]=moleculetoatomindextolabelrdkit[molecule]
                moleculetonumberofatoms[newmolecule]=moleculetonumberofatoms[molecule]
                moleculetoatomindextosymclass[newmolecule]=moleculetoatomindextosymclass[molecule] 
                tempmoleculelist.append(newmolecule)
                moleculetobondconnectivitylist[newmolecule]=moleculetobondconnectivitylist[molecule]
                moleculetoatomindextoatomicsymbol[newmolecule]=moleculetoatomindextoatomicsymbol[molecule]
                moleculetoOBmol[newmolecule]=GenerateMolCopy(moleculetoOBmol[molecule])
                moleculetoatomicpairtodistance[newmolecule]=moleculetoatomicpairtodistance[molecule]
                moleculetoatomindextoinitialposition[newmolecule]=moleculetoatomindextoinitialposition[molecule]
                temp.append(newmolecule)
            moleculessametype.append(temp)


    moleculelist=tempmoleculelist

    return moleculelist,moleculetoatomindextomaxedgesin,moleculetoatomindextomaxedgesout,moleculetoatomindextolabelgraphs,moleculetoatomindextolabelrdkit,moleculetonumberofatoms,moleculetoatomindextosymclass,moleculetobondconnectivitylist,moleculetoatomindextoatomicsymbol,moleculetoOBmol,moleculetoatomicpairtodistance,moleculetoatomindextoinitialposition,moleculessametype


def DetermineAllowedIntermolecularInteractionsMatrix(comb,moleculetonumberofatoms,moleculetoatomindextolabelgraphs,moleculetoatomindextolabelrdkit,moleculetoatomindextoatomicsymbol,moleculetoatomindextosymclass,moleculetoatomindextomaxedgesin,moleculetoatomindextomaxedgesout,atomicsymboltovdwradius,labeltodonoracceptorlabel,atomicsymboltoelectronegativity):
    indextomolecule={}
    moleculetoindexlist={}
    indextoelectronegativity={}
    atomsum=0
    atomnumlist=[]
    for molecule in comb:
        numberofatoms=moleculetonumberofatoms[molecule]
        atomsum+=numberofatoms
        atomnumlist.append(numberofatoms)
    donoracceptormatrix=np.chararray((atomsum,atomsum),itemsize=2) # dont let donors interact with donors or acceptors with acceptors
    intramolecularinteractionsmatrix=np.zeros((atomsum,atomsum)) # dont let intramolecular interactions occur
    intermolecularinteractionsmatrix=np.chararray((atomsum,atomsum),itemsize=7) # keep track of this for computing number of unique graphs
    indextomaxedgesin={}
    indextomaxedgesout={}
    indextoatomlabelgraphs={}
    indextoatomlabelrdkit={}
    indextovdwradius={}
    rowindextoatomindex={}
    for moleculerow in comb:
        moleculerowindex=comb.index(moleculerow)
        currentrowatomsum=atomnumlist[moleculerowindex]
        previousrowatomsum=np.sum(atomnumlist[:moleculerowindex])
        numberofatomsrow=moleculetonumberofatoms[moleculerow]
        atomindextolabelgraphsrow=moleculetoatomindextolabelgraphs[moleculerow]
        atomindextolabelrdkitrow=moleculetoatomindextolabelrdkit[moleculerow]
        atomindextoatomicsymbolrow=moleculetoatomindextoatomicsymbol[moleculerow]
        atomindextosymclassrow=moleculetoatomindextosymclass[moleculerow]
        atomindextomaxedgesin=moleculetoatomindextomaxedgesin[moleculerow]
        atomindextomaxedgesout=moleculetoatomindextomaxedgesout[moleculerow]
        for i in range(numberofatomsrow):
           rowindex=int(i+previousrowatomsum)
           rowlabel=atomindextoatomicsymbolrow[i]
           electronegativity=atomicsymboltoelectronegativity[rowlabel]
           vdwradius=atomicsymboltovdwradius[rowlabel]
           indextovdwradius[rowindex]=vdwradius
           indextoelectronegativity[rowindex]=electronegativity
           rowdonoracceptorlabel=labeltodonoracceptorlabel[rowlabel]
           rowsymclass=atomindextosymclassrow[i]
           maxedgesin=atomindextomaxedgesin[i]
           maxedgesout=atomindextomaxedgesout[i]
           indextomaxedgesin[rowindex]=maxedgesin
           indextomaxedgesout[rowindex]=maxedgesout
           rowclasslabelgraphs=atomindextolabelgraphsrow[i]
           rowclasslabelrdkit=atomindextolabelrdkitrow[i]
           indextoatomlabelgraphs[rowindex]=rowclasslabelgraphs+'-'+str(rowindex+1)+'-'+rowclasslabelrdkit
           indextoatomlabelrdkit[rowindex]=str(rowindex+1)+'-'+rowclasslabelrdkit
           indextomolecule[rowindex]=moleculerow
           rowindextoatomindex[rowindex]=i+1
         
           if moleculerow not in moleculetoindexlist.keys():
               moleculetoindexlist[moleculerow]=[]
           moleculetoindexlist[moleculerow].append(rowindex)
           for moleculecol in comb:
               moleculecolindex=comb.index(moleculecol)
               currentcolatomsum=atomnumlist[moleculecolindex]
               previouscolatomsum=np.sum(atomnumlist[:moleculecolindex])

               numberofatomscol=moleculetonumberofatoms[moleculecol]
               atomindextoatomicsymbolcol=moleculetoatomindextoatomicsymbol[moleculecol]
               atomindextosymclasscol=moleculetoatomindextosymclass[moleculecol]
               for j in range(numberofatomscol):
                   collabel=atomindextoatomicsymbolcol[j]
                   coldonoracceptorlabel=labeltodonoracceptorlabel[collabel]
                   colsymclass=atomindextosymclasscol[j]
                   colindex=int(j+previouscolatomsum)

                   if rowindex<colindex:
                       string=rowdonoracceptorlabel+coldonoracceptorlabel
                       donoracceptormatrix[rowindex,colindex]=string

                       if moleculerow==moleculecol:
                           intramolecularinteractionsmatrix[rowindex,colindex]=1
                       if intramolecularinteractionsmatrix[rowindex,colindex]==0 and (donoracceptormatrix[rowindex,colindex].decode(encoding='UTF-8')=='DA' or donoracceptormatrix[rowindex,colindex].decode(encoding='UTF-8')=='AD'):
                           if rowsymclass>colsymclass:
                               intermolecularinteractionsmatrix[rowindex,colindex]=str(rowsymclass)+'-'+str(colsymclass)
                           else:
                               intermolecularinteractionsmatrix[rowindex,colindex]=str(colsymclass)+'-'+str(rowsymclass)


                       else:
                           intermolecularinteractionsmatrix[rowindex,colindex]=str(0)
                   else:
                       intermolecularinteractionsmatrix[rowindex,colindex]=str(0)
    return atomsum,indextomaxedgesin,indextomaxedgesout,indextoatomlabelgraphs,indextoatomlabelrdkit,indextovdwradius,rowindextoatomindex,intermolecularinteractionsmatrix,moleculetoindexlist,indextomolecule,indextoelectronegativity

def DetermineAllowedIntermolecularInteractions(intermolecularinteractionsmatrix):

    locsintermolecular=np.where(intermolecularinteractionsmatrix!=str(0).encode(encoding='UTF-8'))
    locsintermolecularlist=[]
    for loc in np.transpose(locsintermolecular):
        locsintermolecularlist.append(list(loc))
    intermoleculararray=intermolecularinteractionsmatrix[locsintermolecular]
    return locsintermolecularlist,intermoleculararray

def AtomicPairsToFlattenedIndexLocations(intermoleculararray):
    atompairtobondindex={}
    for i in range(len(intermoleculararray)):
        pair=tuple(intermoleculararray[i])
        if pair not in atompairtobondindex.keys():
            atompairtobondindex[pair]=[]
        atompairtobondindex[pair].append(i)
    return atompairtobondindex


def CreateGraph(array):
    rows, cols = np.where(array == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_nodes_from(np.arange(0,len(array),1))
    gr.add_edges_from(edges)
    return gr


def LabelAllowedIntermolecularInteractions(pairwiselocs,indextomolecule):
    intermolecularinteractionbool={}
    for loc in pairwiselocs:
        first=loc[0]
        second=loc[1]
        firstmolecule=indextomolecule[first]
        secondmolecule=indextomolecule[second]
        key=firstmolecule+'-'+secondmolecule
        revkey=secondmolecule+'-'+firstmolecule
        intermolecularinteractionbool[key]=True
        intermolecularinteractionbool[revkey]=True
    return intermolecularinteractionbool

def CheckMaxEdgesInAndOut(indextoedges,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel):
    viable=True
    indextoedgesin={}
    indextoedgesout={}
    for j in indextoedges.keys():
        maxedgesin=indextomaxedgesin[j]
        maxedgesout=indextomaxedgesout[j]
        connedges=indextoedges[j]
        molecule=indextomolecule[j]

        atomindextosymbol=moleculetoatomindextoatomicsymbol[molecule]
        atomindex=rowindextoatomindex[j]
        label=atomindextosymbol[atomindex-1]
        donoracceptorlabel=labeltodonoracceptorlabel[label]
        if donoracceptorlabel=='A':
            indextoedgesin[j]=connedges
            indextoedgesout[j]=0 
            if connedges>maxedgesin:
                viable=False
                break
        elif donoracceptorlabel=='D':
            indextoedgesin[j]=0
            indextoedgesout[j]=connedges
            if connedges>maxedgesout:
                viable=False
                break
    return viable,indextoedgesin,indextoedgesout



def GrabAllIndexesPairwiseConnected(pairwiselocs,index):
    connectedindexes=[]
    connectedpairs=[]
    for pair in pairwiselocs:
        if index in pair:
            loc=list(pair).index(index)
            if loc==0:
                otherindex=pair[1]
            elif loc==1:
                otherindex=pair[0]
            connectedindexes.append(otherindex)
            connectedpairs.append(pair)
    return connectedindexes,connectedpairs


def RestrictImpossibleAtomicIntermolecularInteractions(indextoedgesout,indextomolecule,intermolecularinteractionbool,pairwiselocs,moleculetoindexlist,indextoneighborindexes,check): 
    # if atom has >1 edgesout the molecules interacting with it must be pairwise connected(check this only if grown all bonds), also atoms on other molecules interacting with that atom can at most have one edgeout
    viable=True
    for index,edgesout in indextoedgesout.items():
        if edgesout>1:
            molecule=indextomolecule[index]
            connectedindexes,connectedpairs=GrabAllIndexesPairwiseConnected(pairwiselocs,index) # currently only assumes two otherwise need to do all combinations
            for i in range(len(connectedindexes)-1):
                j=i+1
                firstidx=connectedindexes[i]
                secondidx=connectedindexes[j]
                firstmolecule=indextomolecule[firstidx]
                secondmolecule=indextomolecule[secondidx]
                firstindexlist=moleculetoindexlist[firstmolecule]
                secondindexlist=moleculetoindexlist[secondmolecule]
                firstneighbs=indextoneighborindexes[firstidx]
                secondneighbs=indextoneighborindexes[secondidx]
                neighbslist=[firstneighbs,secondneighbs]
                for nlist in neighbslist:
                    for index in nlist:
                        if index in indextoedgesout:
                            edgesout=indextoedgesout[index]
                        else:
                            edgesout=0
                        if edgesout>1:
                            viable=False
                
    return viable

def AddBondsAndDonorAcceptorDirection(comb,moleculetobondconnectivitylist,moleculetonumberofatoms,pairwiselocs,indextomolecule,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,array,moleculetoindexlist):
    valuestoaddback=[]
    for moleculeidx in range(len(comb)):
        molecule=comb[moleculeidx]
        indexlist=moleculetoindexlist[molecule]
        minidx=min(indexlist)
        bondconnectivity=moleculetobondconnectivitylist[molecule]
        for bond in bondconnectivity:
            addition=np.add(minidx,np.array(bond))
            valuestoaddback.append(addition)
    grapharray=np.copy(array)
    for bond in valuestoaddback: # add bonds to graph
        grapharray[bond[0],bond[1]]=1

    # now if donor flip over the 1 accross diagonal for better depiction
    for loc in pairwiselocs:
        first=loc[0]
        second=loc[1]
        molecule=indextomolecule[second]
        atomindextosymbol=moleculetoatomindextoatomicsymbol[molecule]
        atomindex=rowindextoatomindex[second]
        label=atomindextosymbol[atomindex-1]
        donoracceptorlabel=labeltodonoracceptorlabel[label]
        if donoracceptorlabel=='D':
            grapharray[second,first]=1
            grapharray[first,second]=0

    return grapharray


def GenerateFlattenedArrayIndexMap(locsintermolecularlist):
    indexpairtobondindex={}
    for i in range(len(locsintermolecularlist)):
        loc=locsintermolecularlist[i]
        indexpairtobondindex[tuple(loc)]=i
    return indexpairtobondindex
       
def AtomTypeToIndexlistAllMolecules(moleculessametype,rowindextoatomindex,moleculetoindexlist,indextomolecule,moleculetoatomindextosymclass,allowedindexes,comb):
    atomtypetoindexlistallmolecules={}
    for moleculessametypearray in moleculessametype:
        atomtypetoindexlist={}
        for molecule in moleculessametypearray:
             if molecule not in comb:
                 continue
             indexlist=moleculetoindexlist[molecule]
             atomindextosymclass=moleculetoatomindextosymclass[molecule]
             for index in indexlist:
                 if index not in allowedindexes:
                     continue
                 atomindex=rowindextoatomindex[index]-1
                 atomtype=atomindextosymclass[atomindex]
                 if atomtype not in atomtypetoindexlist.keys():
                     atomtypetoindexlist[atomtype]=[index]
                 else:
                     atomtypetoindexlist[atomtype].append(index)
        for atomtype,indexlist in atomtypetoindexlist.items():
            if atomtype not in atomtypetoindexlistallmolecules.keys():
                atomtypetoindexlistallmolecules[atomtype]=indexlist
            else:
                atomtypetoindexlistallmolecules[atomtype].extend(indexlist)
    return atomtypetoindexlistallmolecules

def DetermineMoleculeTypes(allowedindexes,atomindextosymtype,moleculetoindexlist,indextomolecule,rowindextoatomindex):
    moleculetonewindexlist={}
    for molecule,indexlist in moleculetoindexlist.items():
        newindexlist=[]
        for index in indexlist:
            if index in allowedindexes:
                newindexlist.append(index)
        moleculetonewindexlist[molecule]=newindexlist
    
   
    moleculetomoleculetype={}
    typematrices=[]
    moleculetype=-1
    for molecule,newindexlist in moleculetonewindexlist.items():
        idxtosymtype={}
        for index in newindexlist:
            symtype=atomindextosymtype[index]
            atomindex=rowindextoatomindex[index]
            idxtosymtype[atomindex]=symtype
        if idxtosymtype not in typematrices:
            typematrices.append(idxtosymtype)
            moleculetype+=1
        moleculetomoleculetype[molecule]=moleculetype
        
    
    moleculetypetoindexlist={}
    for molecule,newindexlist in moleculetonewindexlist.items():
        moleculetype=moleculetomoleculetype[molecule]
        if moleculetype not in moleculetypetoindexlist.keys():
            moleculetypetoindexlist[moleculetype]=newindexlist
        else:
            moleculetypetoindexlist[moleculetype].extend(newindexlist)
            
    
    indextomoleculetypeindexlist={}        
    for index,molecule in indextomolecule.items():
        if index in allowedindexes:
            moleculetype=moleculetomoleculetype[molecule]
            indexlist=moleculetypetoindexlist[moleculetype]
            indextomoleculetypeindexlist[index]=indexlist
        
    
        
    return indextomoleculetypeindexlist
    
    


def FindGroupAutomorphismFingerprint(allowedindexes,atomtypetoindexlistallmolecules,atomindextosymtype,indexpairtobondindex,moleculetoindexlist,indextomolecule,rowindextoatomindex,bondindicesgrown,bondvector):
    newatomtypetoindexlistallmolecules={}
    indextomoleculetypeindexlist=DetermineMoleculeTypes(allowedindexes,atomindextosymtype,moleculetoindexlist,indextomolecule,rowindextoatomindex)
    
    
    for atomtype,indexlist in atomtypetoindexlistallmolecules.items():
        newindexlist=[]
        for index in indexlist:
            if index in allowedindexes:
                newindexlist.append(index)
        length=len(newindexlist)
        if length!=0:
            newatomtypetoindexlistallmolecules[atomtype]=newindexlist

 
    listofindexlists=list(newatomtypetoindexlistallmolecules.values())
    listofindexlists.sort(key = len) 
    assignment=0
    oldindextonewindex={}

    for indexlist in listofindexlists:
        for index in indexlist:
            oldindextonewindex[index]=assignment
            assignment+=1
    
    finalatomtypetoindexlistallmolecules={}
    for atomtype,indexlist in newatomtypetoindexlistallmolecules.items():
        newindexlist=[]
        for index in indexlist:
            newindex=oldindextonewindex[index]
            newindexlist.append(newindex)
        finalatomtypetoindexlistallmolecules[atomtype]=newindexlist
    
    newindextomoleculetypeindexlist={}
    for index,indexlist in indextomoleculetypeindexlist.items():
        newindex=oldindextonewindex[index]
        newindexlist=[]
        
        for index in indexlist:
            newerindex=oldindextonewindex[index]
            newindexlist.append(newerindex)

        newindextomoleculetypeindexlist[newindex]=newindexlist
        
    
    newallowedindexes=[oldindextonewindex[i] for i in allowedindexes]
    atomindextosymtypechoicelist={}

    for index,symtype in atomindextosymtype.items():
        if index in oldindextonewindex.keys():
            newindex=oldindextonewindex[index]
            if newindex in newallowedindexes:
                if symtype in finalatomtypetoindexlistallmolecules.keys():
                    
                    indexlist=finalatomtypetoindexlistallmolecules[symtype]
                    moleculetypeindexlist=newindextomoleculetypeindexlist[newindex]
                    newindexlist=[]
                    for index in indexlist:
                        if index in moleculetypeindexlist:
                            newindexlist.append(index)
                    atomindextosymtypechoicelist[newindex]=newindexlist
                
    choices=[]
    for k, v in sorted(atomindextosymtypechoicelist.items()):
        if v not in choices:
            choices.append(v)
    bondindextoindexpair={v: k for k, v in indexpairtobondindex.items()} 
    atomsets = [set(seq) for seq in choices]
        
    bondindextonewindexpair={}
    for bondindex in bondindicesgrown:
        testindexpair=bondindextoindexpair[bondindex]
        testfirstindex=testindexpair[0]
        testsecondindex=testindexpair[1]
        newtestfirstindex=oldindextonewindex[testfirstindex]
        newtestsecondindex=oldindextonewindex[testsecondindex]
        bondindextonewindexpair[bondindex]=[newtestfirstindex,newtestsecondindex]
    atomindextoatomindicessametype={}
    for atomset in atomsets:
        for atomindex in atomset:
            if atomindex not in atomindextoatomindicessametype.keys():
                atomindextoatomindicessametype[atomindex]=[]
            for otheratomindex in atomset:
                atomindextoatomindicessametype[atomindex].append(otheratomindex) 
    for atomindex,atomindicessametype in atomindextoatomindicessametype.items():
        atomindicessametype=set(atomindicessametype)
        atomindextoatomindicessametype[atomindex]=atomindicessametype

    moleculetonewindexlist={}
    for molecule,indexlist in moleculetoindexlist.items():
        newindexlist=[]
        for index in indexlist:
            if index in oldindextonewindex.keys():
                newindex=oldindextonewindex[index]
                newindexlist.append(newindex)
        moleculetonewindexlist[molecule]=newindexlist
    moleculetomoleculessametype={}
    moleculetomoleculeindex={}
    count=-1
    newindices=[]
    for molecule,newindexlist in moleculetonewindexlist.items():
        if molecule not in moleculetomoleculessametype.keys(): 
            moleculetomoleculessametype[molecule]=[]
        temp=[]
        for idx in newindexlist:
            atomindicessametype=atomindextoatomindicessametype[idx]
            temp.append(atomindicessametype)
        if set(newindexlist) not in newindices:
            newindices.append(set(newindexlist))
            count+=1
        moleculetomoleculeindex[molecule]=count
        for othermolecule,othernewindexlist in moleculetonewindexlist.items():
            othertemp=[]
            for idx in othernewindexlist:
                atomindicessametype=atomindextoatomindicessametype[idx]
                othertemp.append(atomindicessametype)
            if temp==othertemp:
                moleculetomoleculessametype[molecule].append(othermolecule)
    atomindextoatomtype={}
    count=-1
    temp=[]
    for atomindex,atomindicessametype in atomindextoatomindicessametype.items():
        if atomindicessametype not in temp:
            temp.append(atomindicessametype)
            count+=1
        atomindextoatomtype[atomindex]=count
    fingerprint=[]
    bondindextobondtype={}
    newtemp=[]
    count=-1
    for bondindex in bondindicesgrown:
        temp=[]
        newindexpair=bondindextonewindexpair[bondindex]
        for atomindex in newindexpair:
            atomtype=atomindextoatomtype[atomindex]
            temp.append(atomtype)
        if temp not in newtemp and temp[::-1] not in newtemp:
            newtemp.append(temp)
            count+=1
        bondindextobondtype[bondindex]=count
    bondtypetonumberofbondsactive={}
    for bondindex in range(len(bondvector)):
        bondvalue=bondvector[bondindex]
        bondtype=bondindextobondtype[bondindex]    
        if bondtype not in bondtypetonumberofbondsactive.keys():
            bondtypetonumberofbondsactive[bondtype]=0
        if bondvalue==1:
            bondtypetonumberofbondsactive[bondtype]+=1
    for bondtype,numberofbondsactive in bondtypetonumberofbondsactive.items():
        fingerprint.append([bondtype,numberofbondsactive]) 
    return fingerprint,oldindextonewindex,bondindextonewindexpair,atomindextoatomindicessametype,moleculetomoleculessametype,moleculetomoleculeindex,moleculetonewindexlist



           

def CheckForNullInvariant(comb,cols,bondindextoneighborindexes):
    continbool=True
    for i in range(len(cols)):
       column=cols[i]
       column=[int(t) for t in column]
       idx=comb[i]
       allcolumn=False
       neighborindexes=bondindextoneighborindexes[idx]
       for index in column:
           if index in neighborindexes: # then there is a flip
               allcolumn=True
               break
       if allcolumn==False:
           continbool=False
           return continbool

    return continbool
            
def CreateSubImagesMatrix(columnindexes,imagesmatrix):
    cols=[]
    for idx in columnindexes:
        col=imagesmatrix[:,idx]
        cols.append(col)
    cols=np.array(cols)
    return cols


def ConvertIndicesToColumnIndexes(comb,bondindicesgrown):
    columnindexes=[]
    for idx in comb:
        colidx=bondindicesgrown.index(idx)
        columnindexes.append(colidx)
    return columnindexes
 
def FindAllInvariants(indexpairtobondindex,imagesindexmatrix,bondindextoneighborbondindexes,bondindicesgrown,maxgraphinvorder):
    uniqueinvariantsarray=[]
    maxorder=len(bondindicesgrown)
    if maxorder>maxgraphinvorder:
        maxorder=maxgraphinvorder
    for order in range(1,maxorder+1):
        combs=list(combinations(bondindicesgrown,order))
        for comb in combs:
            columnindexes=ConvertIndicesToColumnIndexes(comb,bondindicesgrown)
            cols=CreateSubImagesMatrix(columnindexes,imagesindexmatrix)
            continbool=CheckForNullInvariant(comb,cols,bondindextoneighborbondindexes)
            if continbool==True:
                continue
            uniqueinvariantsarray.append(comb)

    return uniqueinvariantsarray


def ComputeInvariant(matrix):
    colnum=np.size(matrix,1)
    prod=matrix[:,0]
    prod=np.prod(matrix, axis=1)
    invariant=np.sum(prod)
    return  invariant

 
def AllIndexes(moleculetoindexlist):
    allindexes=[]
    for molecule,indexlist in moleculetoindexlist.items():
        for index in indexlist:
            allindexes.append(index)
    return allindexes

def FindNeighborBonds(indexpairtobondindex,indextoneighborindexes):
    bondindextoneighborbondindexes={}
    for indexpair,bondindex in indexpairtobondindex.items():
        first=indexpair[0]
        second=indexpair[1]
        firstneighbs=indextoneighborindexes[first]
        secondneighbs=indextoneighborindexes[second]
        possibleneighbs = list(product(firstneighbs, secondneighbs))
        neighbs=[]
        for neighb in possibleneighbs:
            pair=tuple(neighb)
            rpair=tuple(neighb[::-1])
            nbondidx=None
            if pair in indexpairtobondindex.keys():
                nbondidx=indexpairtobondindex[pair]
            elif rpair in indexpairtobondindex.keys():
                nbondidx=indexpairtobondindex[rpair]
            if nbondidx!=None:
                neighbs.append(nbondidx)
        if len(neighbs)!=0:
            bondindextoneighborbondindexes[bondindex]=neighbs
    return bondindextoneighborbondindexes


def ComputeImageMatrixFromBondVector(bondvector,imagesindexmatrix,bondindicesgrown):
    imagesmatrix=[]
    for i in range(len(imagesindexmatrix)):
        row=imagesindexmatrix[i,:]
        newrow=[]
        append=True
        for j in range(len(row)):
            index=int(row[j])
            value=bondvector[index]
            newrow.append(value)
        imagesmatrix.append(newrow)
    return np.array(imagesmatrix)

def ComputeInvariantArray(bondvector,imagesindexmatrix,uniqueinvariantsarray,bondindicesgrown,bondindextoindexpair):
    invarray=[]
    bondvector=ReorderBondVector(bondindicesgrown,bondvector,bondindextoindexpair)
    imagesmatrix=ComputeImageMatrixFromBondVector(bondvector,imagesindexmatrix,bondindicesgrown)
    for comb in uniqueinvariantsarray:   
        columnindexes=ConvertIndicesToColumnIndexes(comb,bondindicesgrown)
        matrix=CreateSubImagesMatrix(columnindexes,imagesmatrix)
        invariant=ComputeInvariant(matrix)
        invarray.append(invariant)
    return np.array(invarray)

def OrderGraphsBySubGroups(bondvectors,subgroupindextobondvectorindexes,bondvectorlength):
    newbondvectors=[]
    for subgroupindex in range(len(subgroupindextobondvectorindexes.keys())):
        bondvectorindexes=subgroupindextobondvectorindexes[subgroupindex]
        graphs=np.take(bondvectors,bondvectorindexes,axis=0)
        newbondvectors.append(graphs)
    return np.array(newbondvectors)
    

  

def ConvertOldPairsToNewPairs(oldindextonewindex,indexpairtobondindex,allowedindexes):

    newindexpairtobondindex={}
    
    for indexpair,bondindex in indexpairtobondindex.items():
        newpair=[]
        
        for index in indexpair:
            if index in oldindextonewindex.keys():
                newindex=oldindextonewindex[index]
                if newindex in allowedindexes:
                    newpair.append(newindex)
        if len(newpair)==2:
            newindexpairtobondindex[tuple(newpair)]=bondindex

    return newindexpairtobondindex

def InnerGraphIterLoopFingerprint(bondvector,bondvectoridx,allvertices,atomtypetoindexlistallmolecules,atomindextosymtype,indexpairtobondindex,moleculetoindexlist,indextomolecule,rowindextoatomindex,subgroupfingerprints,mappedindices,lastsubgroupindex,subgroupindextobondvectorindexes,subgroupindex,bondindicesgrown,bondindextonewindexpairmaps,atomindextoatomindicessametypemaps,moleculetomoleculessametypemaps,moleculetomoleculeindexmaps,moleculetonewindexlistmaps):
    
    atomsets,oldindextonewindex,bondindextonewindexpair,atomindextoatomindicessametype,moleculetomoleculessametype,moleculetomoleculeindex,moleculetonewindexlist=FindGroupAutomorphismFingerprint(allvertices,atomtypetoindexlistallmolecules,atomindextosymtype,indexpairtobondindex,moleculetoindexlist,indextomolecule,rowindextoatomindex,bondindicesgrown,bondvector)
    
    if bondvectoridx!=0:
        found=np.all(np.isin(atomsets,np.array(subgroupfingerprints)))
    else:
        found=False
    if found==False:
        subgroupfingerprints.append(atomsets)
        mappedindices.append(oldindextonewindex)
        bondindextonewindexpairmaps.append(bondindextonewindexpair)
        atomindextoatomindicessametypemaps.append(atomindextoatomindicessametype)
        moleculetomoleculessametypemaps.append(moleculetomoleculessametype)
        moleculetomoleculeindexmaps.append(moleculetomoleculeindex)
        moleculetonewindexlistmaps.append(moleculetonewindexlist)
        subgroupindex=lastsubgroupindex+1
    lastsubgroupindex=len(subgroupfingerprints)-1
    if subgroupindex not in subgroupindextobondvectorindexes.keys():
        subgroupindextobondvectorindexes[subgroupindex]=[]
    subgroupindextobondvectorindexes[subgroupindex].append(bondvectoridx)

    return subgroupfingerprints,mappedindices,bondindextonewindexpairmaps,lastsubgroupindex,subgroupindextobondvectorindexes,subgroupindex,atomindextoatomindicessametypemaps,moleculetomoleculessametypemaps,moleculetomoleculeindexmaps,moleculetonewindexlistmaps


def cyclic_perm(a):
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b

def cyclic_perm_func(a):
    n = len(a)
    def wrapper(a, n, j):
        def cyc():
            return [a[i - j] for i in range(n)]
        return cyc
    b = [wrapper(a, n, j) for j in range(n)]
    return b

def AllReflectiveAxes(moleculeindicessametype):
    vertices=len(moleculeindicessametype)
    axes=[]
    if (vertices % 2) == 0: # even
        for i in range(len(moleculeindicessametype)):
            vertex=moleculeindicessametype[i]
            trialindex1=i+int(vertices/2)
            trialindex2=np.abs(vertices-trialindex1)
            if trialindex1>vertices-1:
                otherindex=trialindex2
            else:
                otherindex=trialindex1
            othervertex=moleculeindicessametype[otherindex]
            axes.append([vertex,othervertex])
    else: # odd 
        for i in range(len(moleculeindicessametype)):
            vertex=moleculeindicessametype[i]
            axes.append([vertex])
    return axes

def AllReflectivePermutations(moleculeindicessametype):
    vertices=len(moleculeindicessametype)
    perms=[]
    if vertices==4: # square is special case
        a,b,c,d=moleculeindicessametype[:]
        perms.append([b,a,d,c])
        perms.append([d,c,a,b])
        perms.append([a,d,c,b])
        perms.append([c,b,a,d])        
    else:
        axes=AllReflectiveAxes(moleculeindicessametype)   
        for axis in axes:
            indextodistancetoaxis=GenerateDistanceToAxis(axis,moleculeindicessametype)
            indextopermindex=GeneratePermutation(indextodistancetoaxis)
            perm=list(indextopermindex.values())
            perms.append(perm)
    return perms


def GeneratePermutation(indextodistancetoaxis):
    indextopermindex={}
    for index,distance in indextodistancetoaxis.items():
        if distance==0:
            indextopermindex[index]=index
        else:
            for otherindex,otherdistance in indextodistancetoaxis.items():
                if otherdistance==distance and index!=otherindex:
                    indextopermindex[index]=otherindex
    return indextopermindex



def GenerateDistanceToAxis(axis,moleculeindicessametype):
    indextodistancetoaxis={}
    specialvertex=axis[0]
    for i in range(len(moleculeindicessametype)):
        vertex=moleculeindicessametype[i]
        if vertex in axis:
            distance=0
        else:
            dist1=np.abs(vertex-specialvertex)
            dist2=len(moleculeindicessametype)-dist1
            distance=min([dist1,dist2])
        indextodistancetoaxis[vertex]=distance
    return indextodistancetoaxis


def MergePermutations(permslist):
    perms=[]
    indextolistofpermindextoindex={}
    dicsthatmaptosame=[]
    for ls in permslist:
        for i in range(len(ls)):
            subls=ls[i]
            indextopermindex=IndexToPermIndex(subls)
            if len(indextopermindex.keys())==1:
               dicsthatmaptosame.append(indextopermindex) 
            if i not in indextolistofpermindextoindex.keys():
                indextolistofpermindextoindex[i]=[]
            indextolistofpermindextoindex[i].append(indextopermindex)

    for index,listofpermindextoindex in indextolistofpermindextoindex.items():
        for dic in dicsthatmaptosame:
            if dic not in indextolistofpermindextoindex[index]:
                indextolistofpermindextoindex[index].append(dic)


    for index,listofpermindextoindex in indextolistofpermindextoindex.items():
         result = {}
         for d in listofpermindextoindex:
             result.update(d)
         perms.append(list(result.values()))

    return perms

def CombineReflectionsRotations(perms,otherperms):
    for perm in otherperms:
        if perm not in perms:
            perms.append(perm)
    return perms

def MoleculeTypeAutomorphismGroupGenerator(atomindextoatomindicessametype,moleculetomoleculessametype,moleculetomoleculeindex,moleculetonewindexlist,newindexpairtobondindex):
    imagesmatrix=[]
    moleculeindextomolecule={v: k for k, v in moleculetomoleculeindex.items()}
    sametypes=list(moleculetomoleculessametype.values())
    sametypesnumbers=[]
    for ls in sametypes:
        nums=[moleculetomoleculeindex[i] for i in ls]
        sametypesnumbers.append(nums)
    numslist=[]
    rotslist=[]
    refllist=[]
    for ls in sametypesnumbers: # check what do you do if multiple of different types
        if ls not in numslist:
            numslist.append(ls) 
            f = cyclic_perm_func(ls)
            perms=[g() for g in f]
            rotslist.append(perms)
            reflections=AllReflectivePermutations(ls)
            refllist.append(reflections)
    otherperms=MergePermutations(refllist)
    perms=MergePermutations(rotslist)
    perms=CombineReflectionsRotations(perms,otherperms)
    permtoindextopermindex={}
    allatomindices=list(atomindextoatomindicessametype.keys())
    for perm in perms:
        indextopermindex=IndexToPermIndex(perm)
        permtoindextopermindex[tuple(perm)]=indextopermindex
    for perm,indextopermindex in permtoindextopermindex.items():   
        atomindextopermindex={} 
        for index,permindex in indextopermindex.items(): # for every swapped molecule
            indexmol=moleculeindextomolecule[index]
            permmol=moleculeindextomolecule[permindex]
            indexmolatomlist=moleculetonewindexlist[indexmol]
            permmolatomlist=moleculetonewindexlist[permmol]
            for idx in indexmolatomlist:
               indicessametype=atomindextoatomindicessametype[idx]
               for oidx in indicessametype:
                  if oidx in permmolatomlist:
                      if oidx not in atomindextopermindex.values():
                          atomindextopermindex[idx]=oidx
        for atomindex in allatomindices:
            if atomindex not in atomindextopermindex.keys():
                atomindextopermindex[atomindex]=permindex
        bondindextopermindex={}       
        for newindexpair,bondindex in newindexpairtobondindex.items():
            firstpermindex=atomindextopermindex[newindexpair[0]]
            secondpermindex=atomindextopermindex[newindexpair[1]]
            ls=tuple([firstpermindex,secondpermindex])
            if ls in newindexpairtobondindex.keys():
                permbondindex=newindexpairtobondindex[ls]
            elif ls[::-1] in newindexpairtobondindex.keys():
                permbondindex=newindexpairtobondindex[ls[::-1]]
            bondindextopermindex[bondindex]=permbondindex
        bondauto=list(bondindextopermindex.values())
        if bondauto not in imagesmatrix:
            imagesmatrix.append(bondauto)
    return imagesmatrix

def InnerGraphIterLoop(allvertices,indexpairtobondindex,bondindicesgrown,subgroupfingerprints,subgroupimageindexmatrix,indexmaps,atomsets,oldindextonewindex,bondindextonewindexpair,atomindextoatomindicessametype,moleculetomoleculessametype,moleculetomoleculeindex,moleculetonewindexlist):
    newallvertices=[oldindextonewindex[i] for i in allvertices]
    newindexpairtobondindex=ConvertOldPairsToNewPairs(oldindextonewindex,indexpairtobondindex,newallvertices)
    reducedimagesmatrix=MoleculeTypeAutomorphismGroupGenerator(atomindextoatomindicessametype,moleculetomoleculessametype,moleculetomoleculeindex,moleculetonewindexlist,newindexpairtobondindex)

    newbondindextoindexpair={v: k for k, v in newindexpairtobondindex.items()}
    
    subgroupimageindexmatrix.append(reducedimagesmatrix)
    indexmaps.append(newindexpairtobondindex)


    return subgroupimageindexmatrix,indexmaps

def GrowBondVectors(indexpairtobondindex,orderedbondindexlist,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,moleculetoindexlist,moleculetoOBmol,moleculetoatomindextosymclass,moleculessametype,indextoneighborindexes,locsintermolecularlist,atomsum,bondindextoneighborbondindexes,maxgraphinvorder,moleculetobondindices,bondindextotypepair,atomtypetoindexlistallmolecules,atomindextosymtype,prevgraphs,previndexpairtobondindex):
    bondindextoindexpair={v: k for k, v in indexpairtobondindex.items()}

    bondvectormatrix=[]
    maxlength=len(indexpairtobondindex.keys())
    moleculesgrown=[]
    bondindicesgrownmatrix=[]
    bondlengthtolengthofsortedbondvectorgroups={}
    start=len(previndexpairtobondindex.keys())
    for i in tqdm(range(start,maxlength),desc='Growing Graphs'):
        prog=round(((i+1)/maxlength)*100)
        bondindex=orderedbondindexlist[i]
        if i<(maxlength-1):
            bondindicesgrown=orderedbondindexlist[:i+1]
        else:
            bondindicesgrown=orderedbondindexlist[:maxlength]
        if i==start: # initialize first bond
            if start==0:
                bondvectors=np.array([[0],[1]])
            else:
                bondvectors=np.array(prevgraphs)
                bondvectors=GrowOutABond(bondvectors)

        else:
            bondvectors=GrowOutABond(bondvectors)
        maxcheckindex=i+1
        bondlengthtolengthofsortedbondvectorgroups[maxcheckindex]=[]
        allvertices=FindAllVertices(maxcheckindex,bondindextoindexpair,orderedbondindexlist)
        bondvectors,moleculesgrown=CheckGraphRules(bondvectors,bondindextoindexpair,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,moleculetoindexlist,moleculetoOBmol,indextoneighborindexes,maxcheckindex,locsintermolecularlist,atomsum,allvertices,moleculetobondindices,bondindicesgrown,moleculesgrown,prog)
        subgroupfingerprints=[]
        subgroupimageindexmatrix=[]
        indexmaps=[]
        subgroupindextobondvectorindexes={}
        mappedindices=[]
        bondindextonewindexpairmaps=[]
        atomindextoatomindicessametypemaps=[]
        moleculetomoleculessametypemaps=[]
        moleculetomoleculeindexmaps=[]
        lastsubgroupindex=-1
        subgroupindex=0
        moleculetonewindexlistmaps=[]
        for bondvectoridx in tqdm(range(len(bondvectors)),desc='Enumerating Unique Subgroups, GraphLength=%s%%'%prog):
            bondvector=bondvectors[bondvectoridx]
            subgroupfingerprints,mappedindices,bondindextonewindexpairmaps,lastsubgroupindex,subgroupindextobondvectorindexes,subgroupindex,atomindextoatomindicessametypemaps,moleculetomoleculessametypemaps,moleculetomoleculeindexmaps,moleculetonewindexlistmaps=InnerGraphIterLoopFingerprint(bondvector,bondvectoridx,allvertices,atomtypetoindexlistallmolecules,atomindextosymtype,indexpairtobondindex,moleculetoindexlist,indextomolecule,rowindextoatomindex,subgroupfingerprints,mappedindices,lastsubgroupindex,subgroupindextobondvectorindexes,subgroupindex,bondindicesgrown,bondindextonewindexpairmaps,atomindextoatomindicessametypemaps,moleculetomoleculessametypemaps,moleculetomoleculeindexmaps,moleculetonewindexlistmaps)
        subgroupimageindexmatrix,indexmaps=GenerateAllGroupsSerial(allvertices,indexpairtobondindex,bondindicesgrown,subgroupfingerprints,subgroupimageindexmatrix,indexmaps,mappedindices,bondindextonewindexpairmaps,atomindextoatomindicessametypemaps,moleculetomoleculessametypemaps,moleculetomoleculeindexmaps,moleculetonewindexlistmaps)
        subgroupimageindexmatrix=np.array(subgroupimageindexmatrix)
        bondvectors=OrderGraphsBySubGroups(bondvectors,subgroupindextobondvectorindexes,maxcheckindex)
        newbondvectors=[]
        for bondvectorgroupidx in tqdm(range(len(bondvectors)),desc='Iterating over Subgroups, GraphLength=%s'%prog):
            bondvectorgroup=bondvectors[bondvectorgroupidx]
            newindexpairtobondindex=indexmaps[bondvectorgroupidx]
            imageindexmatrix=subgroupimageindexmatrix[bondvectorgroupidx,:]
            uniqueinvariantsarray=FindAllInvariants(newindexpairtobondindex,imageindexmatrix,bondindextoneighborbondindexes,bondindicesgrown,maxgraphinvorder)
            if len(uniqueinvariantsarray)!=0:
                
                invmatrix=ComputeInvariantsArrayForAllGraphs(bondvectorgroup,imageindexmatrix,uniqueinvariantsarray,bondindicesgrown,bondindextoindexpair)
                sortedbondvectorgroups,uniqueinvariantarrays=SortBondVectors(bondvectorgroup,invmatrix)
                if len(sortedbondvectorgroups)==0:
                    raise ValueError('error, no sorting')
                
            else:
                sortedbondvectorgroups=np.array([bondvectorgroup])
            newsortedbondvectorgroups=[]
            for k in tqdm(range(len(sortedbondvectorgroups)),desc='Checking Isomorphisms for group of graphs'):
                sortedbondvectorgroup=sortedbondvectorgroups[k]
                length=len(sortedbondvectorgroup)
                bondlengthtolengthofsortedbondvectorgroups[maxcheckindex].append(length)
                indexestodelete=FindIsomorphicGraphsWithinSubGroup(sortedbondvectorgroup,imageindexmatrix,locsintermolecularlist,atomsum,bondindicesgrown,bondindextoindexpair)
                if len(indexestodelete)!=0:
                    sortedbondvectorgroup=np.delete(sortedbondvectorgroup,indexestodelete,axis=0)
                newsortedbondvectorgroups.extend(sortedbondvectorgroup[:])
            bondvectorgroup=np.array(newsortedbondvectorgroups)   
            newbondvectors.extend(bondvectorgroup[:])
        bondvectors=np.array(newbondvectors)
    return bondvectors,bondlengthtolengthofsortedbondvectorgroups




def GenerateAllGroupsSerial(allvertices,indexpairtobondindex,bondindicesgrown,subgroupfingerprints,subgroupimageindexmatrix,indexmaps,mappedindices,bondindextonewindexpairmaps,atomindextoatomindicessametypemaps,moleculetomoleculessametypemaps,moleculetomoleculeindexmaps,moleculetonewindexlistmaps):
    for i in range(len(subgroupfingerprints)):
        atomsets=subgroupfingerprints[i]
        oldindextonewindex=mappedindices[i]
        bondindextonewindexpair=bondindextonewindexpairmaps[i]
        atomindextoatomindicessametype=atomindextoatomindicessametypemaps[i]
        moleculetomoleculessametype=moleculetomoleculessametypemaps[i]
        moleculetomoleculeindex=moleculetomoleculeindexmaps[i]
        moleculetonewindexlist=moleculetonewindexlistmaps[i]
        subgroupimageindexmatrix,indexmaps=InnerGraphIterLoop(allvertices,indexpairtobondindex,bondindicesgrown,subgroupfingerprints,subgroupimageindexmatrix,indexmaps,atomsets,oldindextonewindex,bondindextonewindexpair,atomindextoatomindicessametype,moleculetomoleculessametype,moleculetomoleculeindex,moleculetonewindexlist)
    
    return subgroupimageindexmatrix,indexmaps


def CheckGraphRules(bondvectors,bondindextoindexpair,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,moleculetoindexlist,moleculetoOBmol,indextoneighborindexes,maxcheckindex,locsintermolecularlist,atomsum,allvertices,moleculetobondindices,bondindicesgrown,moleculesgrown,prog):
    indexestodelete=[]
    for bondvectoridx in tqdm(range(len(bondvectors)),desc='Checking Graph Rules, GraphLength=%s'%prog):
        bondvector=bondvectors[bondvectoridx]
        viablegraph,moleculesgrown=CheckGraph(bondvector,bondindextoindexpair,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,moleculetoindexlist,moleculetoOBmol,indextoneighborindexes,maxcheckindex,locsintermolecularlist,atomsum,allvertices,moleculetobondindices,bondindicesgrown,moleculesgrown)
        if viablegraph==False:
            indexestodelete.append(bondvectoridx)
    bondvectors=np.delete(bondvectors,indexestodelete,0)
    return bondvectors,moleculesgrown

def CheckGraph(bondvector,bondindextoindexpair,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,moleculetoindexlist,moleculetoOBmol,indextoneighborindexes,maxcheckindex,locsintermolecularlist,atomsum,allvertices,moleculetobondindices,bondindicesgrown,moleculesgrown):
    indextoedges=ComputeEdges(bondvector,bondindextoindexpair,bondindicesgrown,atomsum)
    viable,indextoedgesin,indextoedgesout=CheckMaxEdgesInAndOut(indextoedges,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel)
    if viable==False:
        return viable,moleculesgrown
    else:
        check=len(bondvector)==len(bondindextoindexpair.keys())
        pairwiselocs=PairwiseLocationsFromArray(bondvector,bondindextoindexpair)
        intermolecularinteractionbool=LabelAllowedIntermolecularInteractions(pairwiselocs,indextomolecule)
        viable=RestrictImpossibleAtomicIntermolecularInteractions(indextoedgesout,indextomolecule,intermolecularinteractionbool,pairwiselocs,moleculetoindexlist,indextoneighborindexes,check) 
    if viable==True:
        if np.all(bondvector==0)==True and len(bondvector)>1:
            viable=False
        else:
            if check==True:
                moleculesgrown=list(moleculetoindexlist.keys())
                bondvector=ReorderBondVector(bondindicesgrown,bondvector,bondindextoindexpair)
                array=AdjacencyMatrix(bondvector,locsintermolecularlist,atomsum)
                pairwiselocs,newpairwiselocs=PairwiseLocations(array)
                reducemat,molindextomoleculename=ReduceAtomMatrix(array,moleculetoindexlist,moleculesgrown,indextomolecule)
                gr=CreateGraph(reducemat)
                connected=nx.is_connected(gr)
                if connected==False:
                    viable=False
    return viable,moleculesgrown

def ReorderBondVector(bondindicesgrown,bondvector,bondindextoindexpair):
    length=len(bondindextoindexpair.keys())
    reorderedbondvector=np.zeros((length))
    for i in range(len(bondvector)):
        bondindex=bondindicesgrown[i]
        bondvalue=bondvector[i]
        reorderedbondvector[bondindex]=bondvalue
    return reorderedbondvector



def PairwiseLocationsFromArray(bondvector,bondindextoindexpair):
    indexes=np.where(bondvector==1)
    indexes=indexes[0]
    pairwiselocs=[]
    for index in indexes:
        indexpair=bondindextoindexpair[index]
        pairwiselocs.append(indexpair)
    return pairwiselocs


def ComputeEdges(bondvector,bondindextoindexpair,bondindicesgrown,atomsum):
    indextoedges={}
    for i in range(len(bondvector)):
        bondindex=bondindicesgrown[i]
        indexpair=bondindextoindexpair[bondindex]
        value=bondvector[i]
        if value==1:
            for index in indexpair:
                if index in bondindicesgrown:
                    if index not in indextoedges.keys():
                        indextoedges[index]=0
                    indextoedges[index]+=1
    return indextoedges


def GrowOutABond(bondvectors):
    bondvectors=np.append(bondvectors,np.zeros([len(bondvectors),1]),1)
    anotherbondvectors=np.copy(bondvectors)
    anotherbondvectors[:,-1]=1 
    bondvectors=np.append(bondvectors,anotherbondvectors,0)
    return bondvectors

def ComputeInvariantsArrayForAllGraphs(bondvectors,imagesindexmatrix,uniqueinvariantsarray,bondindicesgrown,bondindextoindexpair):
    invmatrix=[]
    for bondvector in bondvectors:
        invarray=ComputeInvariantArray(bondvector,imagesindexmatrix,uniqueinvariantsarray,bondindicesgrown,bondindextoindexpair)  
        invmatrix.append(invarray)
    invmatrix=np.array(invmatrix,dtype=int)
    return invmatrix
 
 
def SortBondVectors(bondvectors,sortmatrix):
    shape=sortmatrix.shape
    dim=shape[0]
    if dim>1:
        unq, count = np.unique(sortmatrix, axis=0, return_counts=True)
    else:
        unq=sortmatrix[0,:]
    bondvectormatrix=[]
    for arr in unq:
        indexes=np.where((sortmatrix == arr).all(axis=1))
        graphs=bondvectors[indexes]
        check=np.all(np.isin(graphs,bondvectormatrix))
        dim=graphs.shape[0]
        if check==False and dim!=0:
            bondvectormatrix.append(graphs)
    if len(bondvectormatrix)==0:
        bondvectormatrix=[bondvectors[:]]
    return np.array(bondvectormatrix),unq


def FindIsomorphicGraphsWithinSubGroup(sortedbondvectorgroup,imageindexmatrix,locsintermolecularlist,atomsum,bondindicesgrown,bondindextoindexpair):
    indexestodelete=[]
    uniquebondvectors=[] 
    for i in tqdm(range(len(sortedbondvectorgroup)),desc='Checking for Graph Isomorphisms'):
        bondvector=sortedbondvectorgroup[i,:]
        foundiso=False
        uniquevecs=np.array(uniquebondvectors)
        for j in range(len(uniquevecs)):
            comparevector=uniquevecs[j,:]
            foundiso=CheckIsomorphism(bondvector,comparevector,locsintermolecularlist,atomsum,bondindicesgrown,bondindextoindexpair)

            if foundiso==True:
                indexestodelete.append(i)
                break
        if foundiso==False:
            uniquebondvectors.append(bondvector)
    return np.array(indexestodelete)



def CheckIsomorphism(firstbondvector,secondbondvector,locsintermolecularlist,atomsum,bondindicesgrown,bondindextoindexpair):
    firstbondvector=ReorderBondVector(bondindicesgrown,firstbondvector,bondindextoindexpair)
    secondbondvector=ReorderBondVector(bondindicesgrown,secondbondvector,bondindextoindexpair)
    firstmatrix=AdjacencyMatrix(firstbondvector,locsintermolecularlist,atomsum)
    secondmatrix=AdjacencyMatrix(secondbondvector,locsintermolecularlist,atomsum)
    gr1=CreateGraph(firstmatrix)
    gr2=CreateGraph(secondmatrix)
    return nx.is_isomorphic(gr1,gr2)
 
       

def GenerateAllowedIndexes(indexpairtobondindex,allindexes):
    allowedindexes=np.array([])
    for index in allindexes:
        found=False
        for pair in indexpairtobondindex.keys():
            if index in pair:
                found=True
        if found==True:
            allowedindexes=np.append(allowedindexes,index)
    return allowedindexes


def IndexToPermIndex(groupoperation):
    currentindex=np.amin(groupoperation)
    indextopermindex={}
    for j in range(len(groupoperation)):
        permindex=groupoperation[j]
        indextopermindex[currentindex]=permindex
        currentindex+=1
    return indextopermindex


def FindAllVertices(bondvectorlength,bondindextoindexpair,orderedbondindexvector):
    allvertices=np.array([],dtype=int)
    for i in range(bondvectorlength):
        bondindex=orderedbondindexvector[i]
        indexpair=bondindextoindexpair[bondindex]
        for index in indexpair:
            if np.all(np.isin(index,allvertices))==False:
                allvertices=np.append(allvertices,index)
    return allvertices



def GenerateBondIndexToTypePair(indexpairtobondindex,moleculetoatomindextosymclass,indextomolecule,rowindextoatomindex):
    bondindextotypepair={}
    for indexpair,bondindex in indexpairtobondindex.items():
        typepair=[]
        for index in indexpair:
            molecule=indextomolecule[index] 
            atomindextosymclass=moleculetoatomindextosymclass[molecule]
            atomindex=rowindextoatomindex[index]-1
            symclass=atomindextosymclass[atomindex]
            typepair.append(symclass)
        bondindextotypepair[bondindex]=set(typepair)
    return bondindextotypepair

def GenerateAtomIndexToSymType(moleculetoatomindextosymclass,indextomolecule,rowindextoatomindex):
    atomindextosymtype={}
    for index,molecule in indextomolecule.items():
        atomindex=rowindextoatomindex[index]-1
        atomindextosymclass=moleculetoatomindextosymclass[molecule]
        symtype=atomindextosymclass[atomindex]
        atomindextosymtype[index]=symtype
    return atomindextosymtype


def FilterCombinations(combs,moleculelist):
    newcombs = [] 
    combstried=[]
    combtried=InnerMoleculeNameArray(moleculelist)
    moleculelistset=set(combtried)
    for comb in combs:
        combtried=InnerMoleculeNameArray(comb)
        combtriedset=set(combtried)
        if combtried not in combstried and len(moleculelistset)==len(combtriedset):
            combstried.append(combtried)
            newcombs.append(comb)
    return newcombs 


def InnerMoleculeNameArray(comb):
    combtried=[]
    for molname in comb:
        filenamesplit=molname.split('.')
        innerfile=filenamesplit[0]
        molnamesplit=innerfile.split('_')
        truemolname=molnamesplit[0]
        combtried.append(truemolname)
    return combtried


def GetCombinations(n,moleculenametodic,maxgraphinvorder,prevgraphs,moltypestoclssizetoindextobondindex):
    moleculetoatomindextomaxedgesin=moleculenametodics['atomindextomaxedgesin']
    moleculetoatomindextomaxedgesout=moleculenametodics['atomindextomaxedgesout']
    moleculetoatomindextolabelgraphs=moleculenametodics['atomindextolabelgraphs']
    moleculetoatomindextolabelrdkit=moleculenametodics['atomindextolabelrdkit']
    moleculetonumberofatoms=moleculenametodics['numberofatoms']
    moleculetoatomindextosymclass=moleculenametodics['atomindextosymclass']
    moleculenames=moleculenametodics['moleculenames']
    moleculetobondconnectivitylist=moleculenametodics['bondconnectivitylist']
    moleculetoatomindextoatomicsymbol=moleculenametodics['atomindextoatomicsymbol']
    moleculetoOBmol=moleculenametodics['OBmol']
    moleculetoOBmolbackup={}
    moleculetoatomicpairtodistance=moleculenametodics['atomicpairtodistance']
    labeltodonoracceptorlabel=moleculenametodics['labeltodonoracceptorlabel']
    moleculetoatomindextoinitialposition=moleculenametodics['atomindextoinitialposition']
    atomicsymboltovdwradius=moleculenametodics['atomicsymboltovdwradius']
    atomicsymboltoelectronegativity=moleculenametodics['atomicsymboltoelectronegativity']

    moleculelist,moleculetoatomindextomaxedgesin,moleculetoatomindextomaxedgesout,moleculetoatomindextolabelgraphs,moleculetoatomindextolabelrdkit,moleculetonumberofatoms,moleculetoatomindextosymclass,moleculetobondconnectivitylist,moleculetoatomindextoatomicsymbol,moleculetoOBmol,moleculetoatomicpairtodistance,moleculetoatomindextoinitialposition,moleculessametype=DuplicateMoleculesAndMolecularPropertyDictionaries(moleculenames,n,moleculetoatomindextomaxedgesin,moleculetoatomindextomaxedgesout,moleculetoatomindextolabelgraphs,moleculetoatomindextolabelrdkit,moleculetonumberofatoms,moleculetoatomindextosymclass,moleculetobondconnectivitylist,moleculetoatomindextoatomicsymbol,moleculetoOBmol,moleculetoatomicpairtodistance,moleculetoatomindextoinitialposition)
    
    combs = combinations(moleculelist,n) # make matrices with these combs
    combs = FilterCombinations(combs,moleculelist)
    digraphmatarray=[] # store all matrices for every combination here
    graphmatarray=[]
    graphlabels=[]
    rdkitlabels=[]
    indextomoleculearray=[] # keep track of which molecules used for each matrix
    moleculetoindexlistarray=[]
    reducematarray=[]
    reducelabelsarray=[]
    rowindextoatomindexarray=[]
    indextovdwradiusarray=[]
    indextoelectronegativityarray=[]
    combarray=[]
    bondlengthtolengthofsortedbondvectorgroupsarray=[]
    for comb in combs:
        print('comb',comb,'n',n)
        print('*********************************************************')
        combarray.append(comb)
        atomsum,indextomaxedgesin,indextomaxedgesout,indextoatomlabelgraphs,indextoatomlabelrdkit,indextovdwradius,rowindextoatomindex,intermolecularinteractionsmatrix,moleculetoindexlist,indextomolecule,indextoelectronegativity=DetermineAllowedIntermolecularInteractionsMatrix(comb,moleculetonumberofatoms,moleculetoatomindextolabelgraphs,moleculetoatomindextolabelrdkit,moleculetoatomindextoatomicsymbol,moleculetoatomindextosymclass,moleculetoatomindextomaxedgesin,moleculetoatomindextomaxedgesout,atomicsymboltovdwradius,labeltodonoracceptorlabel,atomicsymboltoelectronegativity)
        locsintermolecularlist,intermoleculararray=DetermineAllowedIntermolecularInteractions(intermolecularinteractionsmatrix)
        allindexes=AllIndexes(moleculetoindexlist)
        neighborindexes,indextoneighborindexes=GrabNeighborIndexes(allindexes,indextomolecule,moleculetoindexlist,moleculetoOBmol,rowindextoatomindex)
        indexpairtobondindex=GenerateFlattenedArrayIndexMap(locsintermolecularlist)
        bondindextotypepair=GenerateBondIndexToTypePair(indexpairtobondindex,moleculetoatomindextosymclass,indextomolecule,rowindextoatomindex)
        atomindextosymtype=GenerateAtomIndexToSymType(moleculetoatomindextosymclass,indextomolecule,rowindextoatomindex) 
        bondindextoneighborbondindexes=FindNeighborBonds(indexpairtobondindex,indextoneighborindexes)
        allowedindexes=GenerateAllowedIndexes(indexpairtobondindex,allindexes)
        atomtypetoindexlistallmolecules=AtomTypeToIndexlistAllMolecules(moleculessametype,rowindextoatomindex,moleculetoindexlist,indextomolecule,moleculetoatomindextosymclass,allowedindexes,comb)
        bondvector=np.ones(len(indexpairtobondindex.keys()))

        orderedbondindexlist,indexpairtobondindex,moltypestoclssizetoindextobondindex,previndexpairtobondindex=ReorderTheBondIndexList(moltypestoclssizetoindextobondindex,indexpairtobondindex,comb,n)    
        bondindextoindexpair={v: k for k, v in indexpairtobondindex.items()}
        moleculetobondindices=GenerateMoleculeToBondIndices(bondindextoindexpair,moleculetoindexlist)

        bondvectormatrix,bondlengthtolengthofsortedbondvectorgroups=GrowBondVectors(indexpairtobondindex,orderedbondindexlist,indextomolecule,indextomaxedgesin,indextomaxedgesout,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,moleculetoindexlist,moleculetoOBmol,moleculetoatomindextosymclass,moleculessametype,indextoneighborindexes,locsintermolecularlist,atomsum,bondindextoneighborbondindexes,maxgraphinvorder,moleculetobondindices,bondindextotypepair,atomtypetoindexlistallmolecules,atomindextosymtype,prevgraphs,previndexpairtobondindex)
        bondlengthtolengthofsortedbondvectorgroupsarray.append(bondlengthtolengthofsortedbondvectorgroups)
        moleculesgrown=list(moleculetoindexlist.keys())
        bondindicesgrown=orderedbondindexlist
        reorderedbondvectormatrix=[]
        for bondvectoridx in range(len(bondvectormatrix)):
            append=False
            bondvector=bondvectormatrix[bondvectoridx]
            #bondvector=ReorderBondVector(bondindicesgrown,bondvector,bondindextoindexpair)
            reorderedbondvectormatrix.append(bondvector)
            array=AdjacencyMatrix(bondvector,locsintermolecularlist,atomsum)
            pairwiselocs,newpairwiselocs=PairwiseLocations(array)
            grapharray=AddBondsAndDonorAcceptorDirection(comb,moleculetobondconnectivitylist,moleculetonumberofatoms,pairwiselocs,indextomolecule,moleculetoatomindextoatomicsymbol,rowindextoatomindex,labeltodonoracceptorlabel,array,moleculetoindexlist)
            reducemat,molindextomoleculename=ReduceAtomMatrix(grapharray,moleculetoindexlist,moleculesgrown,indextomolecule)
            
            graphmatarray.append(np.array(grapharray))
            graphlabels.append(indextoatomlabelgraphs)
            rdkitlabels.append(indextoatomlabelrdkit)
            indextomoleculearray.append(indextomolecule)
            rowindextoatomindexarray.append(rowindextoatomindex)
            moleculetoindexlistarray.append(moleculetoindexlist)
            reducematarray.append(reducemat)
            reducelabelsarray.append(molindextomoleculename)
            indextovdwradiusarray.append(indextovdwradius)
            indextoelectronegativityarray.append(indextoelectronegativity)
    for molecule in moleculetoOBmol.keys():
        moleculetoOBmolbackup[molecule]=GenerateMolCopy(moleculetoOBmol[molecule])
    # add back to nested dictionary with copies
    moleculenametodics['atomindextomaxedgesin']=moleculetoatomindextomaxedgesin
    moleculenametodics['atomindextomaxedgesout']=moleculetoatomindextomaxedgesout
    moleculenametodics['atomindextolabelgraphs']=moleculetoatomindextolabelgraphs
    moleculenametodics['numberofatoms']=moleculetonumberofatoms
    moleculenametodics['atomindextosymclass']=moleculetoatomindextosymclass
    moleculenametodics['moleculenames']=moleculenames
    moleculenametodics['bondconnectivitylist']=moleculetobondconnectivitylist
    moleculenametodics['atomindextoatomicsymbol']=moleculetoatomindextoatomicsymbol
    moleculenametodics['OBmol']=moleculetoOBmol
    moleculenametodics['OBmolbackup']=moleculetoOBmolbackup
    moleculenametodics['atomicpairtodistance']=moleculetoatomicpairtodistance
    moleculenametodics['atomindextoinitialposition']=moleculetoatomindextoinitialposition
    return graphmatarray,reducematarray,reducelabelsarray,indextomoleculearray,rowindextoatomindexarray,moleculetoindexlistarray,graphlabels,rdkitlabels,moleculenametodics,indextovdwradiusarray,indextoelectronegativityarray,combarray,bondlengthtolengthofsortedbondvectorgroupsarray,moltypestoclssizetoindextobondindex,reorderedbondvectormatrix

                 
def GenerateMolTypesFromComb(comb):
    moltypes=[]
    for molecule in comb:
        molsplit=molecule.split('_')
        header=molsplit[0]
        if header not in moltypes:
            moltypes.append(header)
    return tuple(set(moltypes))

def ReorderTheBondIndexList(moltypestoclssizetoindextobondindex,indexpairtobondindex,comb,clssize):
    moltypes=GenerateMolTypesFromComb(comb)
    prevclssize=clssize-1
    if moltypes not in moltypestoclssizetoindextobondindex.keys():
        moltypestoclssizetoindextobondindex[moltypes]={}
    clssizetoindextobondindex=moltypestoclssizetoindextobondindex[moltypes]
    if len(clssizetoindextobondindex.keys())==0:
        previndexpairtobondindex={}
    else:
        if prevclssize not in clssizetoindextobondindex.keys():
            previndexpairtobondindex={}
        else:
            previndexpairtobondindex=clssizetoindextobondindex[prevclssize]

    newindexpairtobondindex=previndexpairtobondindex.copy()
    if len(previndexpairtobondindex.keys())==0:
        orderedbondindexlist=list(indexpairtobondindex.values())
    else:
        startindex=len(previndexpairtobondindex.keys())
        count=startindex
        for indexpair,bondindex in indexpairtobondindex.items():
            if indexpair not in newindexpairtobondindex.keys():
                newindexpairtobondindex[indexpair]=count
                count+=1
        orderedbondindexlist=list(newindexpairtobondindex.values())
        indexpairtobondindex=newindexpairtobondindex
    if clssize not in clssizetoindextobondindex.keys():
        clssizetoindextobondindex[clssize]=indexpairtobondindex
        moltypestoclssizetoindextobondindex[moltypes]=clssizetoindextobondindex

    return orderedbondindexlist,indexpairtobondindex,moltypestoclssizetoindextobondindex,previndexpairtobondindex


def GenerateMoleculeToBondIndices(bondindextoindexpair,moleculetoindexlist):
    moleculetobondindices={}
    for molecule,indexlist in moleculetoindexlist.items():
        bondindices=FindBondIndicesFromVertices(indexlist,bondindextoindexpair)
        moleculetobondindices[molecule]=bondindices
    return moleculetobondindices
             
def FindBondIndicesFromVertices(indexlist,bondindextoindexpair):
    bondindices=[]
    for bondindex,indexpair in bondindextoindexpair.items():
        for index in indexpair:
            if index in indexlist and bondindex not in bondindices:
                bondindices.append(bondindex)
    return bondindices 


def Chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def ChunksList(gen):
    newlst=[]
    for item in gen:
        newlst.append(item)
    return newlst


def PlotMatrices(matrices,mylabels,molsperrow,outputfilepath,basename,digraph=True,reduced=False):
    molsPerImage=molsperrow**2
    matchunks=ChunksList(Chunks(matrices,molsPerImage))
    labchunks=ChunksList(Chunks(mylabels,molsPerImage))
    prevmatslen=len(matchunks[0])
    plt.close('all')
    for i in tqdm(range(len(matchunks)),desc='Plotting Matrices'):
        mats=matchunks[i]
        labs=labchunks[i]
        fig, axes = plt.subplots(nrows=molsperrow, ncols=molsperrow)
        ax = axes.flatten()
        for j in range(len(mats)):
            mat=mats[j]
            curlabs=labs[j]
            bondednodelists=[]
            bondednodelistsforcolors=[]
            pairwiselocs=[]
            alllocs=[]
            for k in range(len(mat)):
                for l in range(len(mat)):
                    temp=[k,l]
                    if mat[k,l]==1:
                        alllocs.append(temp)
                    if mat[k,l]==1 and mat[l,k]==1:
                        if temp not in bondednodelists and temp[::-1] not in bondednodelists:
                            bondednodelists.append(tuple(temp))
                            bondednodelistsforcolors.append(tuple(temp))
                            bondednodelists.append(tuple(temp[::-1]))
                    else:
                        if k!=l:
                            if mat[k,l]==1: 
                                pairwiselocs.append(tuple(temp))
            nodeidxtocolortuple={}
            idxs=0
            listofsamecolorednodes=[]
            if reduced==False:
                for bond in bondednodelistsforcolors:
                    temp=[]
                    for nodeidx in bond:
                        nodeidxtocolortuple[nodeidx]=idxs
                        temp.append(nodeidx)
                    listofsamecolorednodes.append(temp)
                    idxs+=1
            else:
                for k in range(len(mat)):
                    temp=[k]
                    nodeidxtocolortuple[k]=idxs
                    idxs+=1
                    listofsamecolorednodes.append(temp)
            indextocolor={0:'red',1:'green',2:'blue',3:'yellow',4:'cyan',5:'magenta',6:'black',7:'white'}
            colormap={}
            for index in range(len(mat)):
                coloridx=nodeidxtocolortuple[index]
                if coloridx not in indextocolor.keys():
                    newcoloridx=coloridx % len(indextocolor.keys())
                else:
                    newcoloridx=coloridx
                color=indextocolor[newcoloridx]
                colormap[index]=color
            if digraph==True:
                gr = nx.DiGraph()
            else:
                gr=nx.Graph()
            for k in range(len(mat)):
                gr.add_node(k)
            pos=nx.spring_layout(gr)

            for sublist in listofsamecolorednodes:
                color=colormap[sublist[0]]
                nx.draw_networkx_nodes(gr,pos,nodelist=sublist,node_color=color,ax=ax[j])
            
            if reduced==False:
                nx.draw_networkx_edges(gr,pos,edgelist=pairwiselocs,ax=ax[j])
                nx.draw_networkx_edges(gr,pos,edgelist=bondednodelists,ax=ax[j],edge_color='r')
            else:
                nx.draw_networkx_edges(gr,pos,edgelist=alllocs,ax=ax[j])

            nx.draw_networkx_labels(gr,pos,labels=curlabs,ax=ax[j],font_size=10)
            ax[j].set_axis_off()
        for j in range(len(mats),molsPerImage):
            fig.delaxes(axes.flatten()[j])
        if i>0:
            factor=1
        else:
            factor=0
        firstj=i*prevmatslen+factor
        secondj=firstj+len(mats)-factor
        prevmatslen=len(mats)
        plt.show()
        plt.savefig(basename+'_'+str(firstj)+'-'+str(secondj)+'.png')


def PlotScalingAndUniqueGraphsAndStructuresAllSizes(maxmatsize,moleculenametodics,molsperrow,outputfileformat,pairwisedistanceratio,outputfilepath,stericcutoff,maxgraphinvorder,molecules):
    totalgrapharray=[]
    totalindextomoleculearray=[]
    narray=[]
    timearray=[]
    graphlabelsarray=[] 
    rdkitlabelsarray=[]
    totalrowindextoatomindexarray=[]
    totalmoleculetoindexlistarray=[]
    reducedtotalgrapharray=[]
    reducedtotallabels=[]
    totalindextovdwradiusarray=[]
    totalindextoelectronegativityarray=[]
    totalcombsarray=[]
    totalbondlengthtolengthofsortedbondvectorgroupsarray=[]
    reorderedbondvectormatrix=[]
    moltypestoclssizetoindextobondindex={}
    if len(molecules)==1:
        bgnclssize=2
    else:
        bgnclssize=len(molecules)
    for matsize in tqdm(range(bgnclssize,maxmatsize+1),desc='Enumerating Graphs'):
        start=time.time()
        grapharray,reducematarray,reducelabelsarray,indextomoleculearray,rowindextoatomindexarray,moleculetoindexlistarray,graphlabels,rdkitlabels,moleculenametodics,indextovdwradiusarray,indextoelectronegativityarray,combarray,bondlengthtolengthofsortedbondvectorgroupsarray,moltypestoclssizetoindextobondindex,reorderedbondvectormatrix=GetCombinations(matsize,moleculenametodics,maxgraphinvorder,reorderedbondvectormatrix,moltypestoclssizetoindextobondindex)
        end=time.time()
        exec=end-start
        timearray.append(exec)
        narray.append(matsize)
        totalcombsarray.append(combarray)
        totalbondlengthtolengthofsortedbondvectorgroupsarray.append(bondlengthtolengthofsortedbondvectorgroupsarray)
        if len(grapharray)>1:
            totalgrapharray.extend(grapharray)
            reducedtotalgrapharray.extend(reducematarray)
            reducedtotallabels.extend(reducelabelsarray)
        totalindextomoleculearray.extend(indextomoleculearray)
        totalrowindextoatomindexarray.extend(rowindextoatomindexarray)
        totalmoleculetoindexlistarray.extend(moleculetoindexlistarray)
        totalindextovdwradiusarray.extend(indextovdwradiusarray)
        totalindextoelectronegativityarray.extend(indextoelectronegativityarray)
        graphlabelsarray=graphlabelsarray+graphlabels
        rdkitlabelsarray=rdkitlabelsarray+rdkitlabels
    PlotTimeScaling(timearray,narray,outputfilepath,'GraphEnumerationTimeScaling.png','Matrix Size vs Enumeration Time')
    if len(totalgrapharray)>0:
        PlotMatrices(totalgrapharray,graphlabelsarray,molsperrow,outputfilepath,'Graphs',digraph=True,reduced=False)
    if len(reducedtotalgrapharray)>0:
        PlotMatrices(reducedtotalgrapharray,reducedtotallabels,molsperrow,outputfilepath,'Graphs_Reduced',digraph=True,reduced=True)
    WriteOutMatrices(totalgrapharray,'Structure',outputfilepath)
    WriteOutDictionaries(graphlabelsarray,'GraphLabels.txt')
    PlotSortedGraphsSizes(totalbondlengthtolengthofsortedbondvectorgroupsarray,totalcombsarray,narray)
    filenamearray,neighborindexes,originalcoordsarray,narray,timearray=Generate3DStructures(totalgrapharray,totalindextomoleculearray,totalmoleculetoindexlistarray,totalrowindextoatomindexarray,totalindextovdwradiusarray,totalindextoelectronegativityarray,moleculenametodics,outputfileformat,pairwisedistanceratio,stericcutoff,outputfilepath,'Structure',dontmove=False,inputcoordinates=None)
    PlotTimeScaling(timearray,narray,outputfilepath,'StructureGenerationTimeScaling.png','Matrix Size vs Structure Generation Time')

    WriteOutDictionaries(neighborindexes,'NeighborIndexes.txt')

def PlotSortedGraphsSizes(totalbondlengthtolengthofsortedbondvectorgroupsarray,totalcombsarray,narray):
    for i in range(len(totalcombsarray)):
        combs=totalcombsarray[i]
        bondlengthtolengthofsortedbondvectorgroupsarray=totalbondlengthtolengthofsortedbondvectorgroupsarray[i] 
        clustersize=narray[i]
        for j in range(len(combs)):
            comb=combs[j]
            bondlengthtolengthofsortedbondvectorgroups=bondlengthtolengthofsortedbondvectorgroupsarray[j]
            bondlengths,averagegrouplengths,totalgroups=GrabAverageGroupLengths(bondlengthtolengthofsortedbondvectorgroups)
            PlotAverageGroupLengths(bondlengths,averagegrouplengths,totalgroups,clustersize,combs)

def PlotAverageGroupLengths(bondlengths,averagegrouplengths,totalgroups,clustersize,comb):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    line1, =ax.plot(bondlengths,averagegrouplengths,'bo',label='AvgGrpLengths')
    ax2=ax.twinx()
    line2, =ax2.plot(bondlengths,totalgroups,'ro',label='TotalGrps')
    ax2.set_ylabel("Total Sorted SubGroups")
    plt.legend(handles=[line1,line2],loc='best')


    ax.set_ylabel('Average Invariant Sorted Group Lengths')
    ax.set_xlabel('Growing Bond Array Length')
    combstr=''
    for el in comb:
       for newel in el:
           newel=newel.split('.')[0]
           combstr+=str(newel)+'-'
    title=('Clustersize = %s, Combs=%s'%(str(clustersize),combstr))
    filename='AverageInvariantSortedGroupLengthsVsGrowingBondArrayLength-Size-i%s-Combs-%s'%(str(clustersize),combstr)
    plt.title(title)
    plt.show()
    plt.savefig(filename)




def GrabAverageGroupLengths(bondlengthtolengthofsortedbondvectorgroups):
    bondlengths=[]
    averagegrouplengths=[]    
    totalgroups=[]
    for bondlength,lengthofsortedbondvectorgroups in bondlengthtolengthofsortedbondvectorgroups.items():
        if len(lengthofsortedbondvectorgroups)!=0:
            mean=np.mean(lengthofsortedbondvectorgroups)
            bondlengths.append(bondlength)
            averagegrouplengths.append(mean)
            totalgroups.append(len(lengthofsortedbondvectorgroups))

    return bondlengths,averagegrouplengths,totalgroups

def WriteOutDictionaries(array,name):
    with open(name, 'w') as fout:
        json.dump(array, fout)

def WriteOutMatrices(matrices,basename,outputfilepath):
    for i in range(len(matrices)):
        mat=matrices[i]
        fname=basename+'_'+str(i)+'.txt'
        np.savetxt(fname,mat)

def PlotTimeScaling(timearray,narray,outputfilepath,filename,title):
    plt.plot(timearray, narray)
    plt.xlabel('Time (s)')
    plt.ylabel('Matrix Size (nxn)')
    plt.title(title)
    plt.show()
    plt.savefig(filename)

def GrabIndexToCoordinates(mol):
    indextocoordinates={}
    iteratom = openbabel.OBMolAtomIter(mol)
    for atom in iteratom:
        atomidx=atom.GetIdx()
        rdkitindex=atomidx-1
        coords=[atom.GetX(),atom.GetY(),atom.GetZ()]
        indextocoordinates[rdkitindex]=coords
    return indextocoordinates

def AddInputCoordinatesAsDefaultConformer(m,indextocoordinates):
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        x,y,z = indextocoordinates[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    return m 


def CheckIfPairwiseInteractionIsBackAndForth(pair,pairwiselocs,indextomolecule,indextoneighborindexes):
    backandforthpairwise=False
    firstindex=pair[0]
    firstmolecule=indextomolecule[firstindex]
    secondindex=pair[1]
    neighbors=indextoneighborindexes[secondindex]
    backandforthpair=None
    for nindex in neighbors:
        for otherpair in pairwiselocs:
            if set(pair)!=set(otherpair):
                firstotherindex=otherpair[0]
                if nindex==firstotherindex:
                    secondotherindex=otherpair[1]
                    secondmolecule=indextomolecule[secondotherindex]
                    if firstmolecule==secondmolecule:
                        backandforthpairwise=True
                        backandforthpair=otherpair
    return backandforthpairwise,backandforthpair

def GenerateReferenceAngles(pairwiselocs,indextomolecule,indextoneighborindexes,targetangle):
    indicestoreferenceangle={}
    indicestobackandforthindices={}
    for pair in pairwiselocs:
        donorindex=pair[0]
        acceptorindex=pair[1]
        donorneighbs=indextoneighborindexes[donorindex]
        backandforthpairwise,backandforthpair=CheckIfPairwiseInteractionIsBackAndForth(pair,pairwiselocs,indextomolecule,indextoneighborindexes)
        if backandforthpairwise==True:
            angle=targetangle
        else:
            angle=0
        if len(donorneighbs)==1:
            donorneighbindex=donorneighbs[0]
            angleindices=tuple([donorindex,donorneighbindex,acceptorindex])
            if backandforthpairwise==True:
                acceptorneighbindex=backandforthpair[0]
                backandforthindices=tuple([acceptorneighbindex,acceptorindex,donorneighbindex])
        indicestoreferenceangle[angleindices]=angle
        if backandforthpairwise==True:
            indicestobackandforthindices[angleindices]=backandforthindices

    return indicestoreferenceangle,indicestobackandforthindices


def PairwiseLocations(mat):
    pairwiselocs=[]
    locs=np.transpose(np.where(mat==1))

    for loc in locs:
        i=loc[0]
        j=loc[1]
        if mat[j,i]==1: # then bond and dont append
            continue
        else:
            pairwiselocs.append(loc)
    newpairwiselocs=[]
    for pair in pairwiselocs:
        newpairwiselocs.append(list(pair))
    return pairwiselocs,newpairwiselocs

def GrabNeighborIndexes(givenindexlist,indextomolecule,moleculetoindexlist,moleculetoOBmol,rowindextoatomindex):

    neighborindexes=[]
    indextoneighborindexes={} # need for getting donor bond direction
    for index in givenindexlist:
        molecule=indextomolecule[index]
        indexlist=moleculetoindexlist[molecule]
        mol=moleculetoOBmol[molecule]
        neighbs=[]
        if index not in indextoneighborindexes.keys():
            for i in indexlist:
                if i!=index:
                    babelindex=rowindextoatomindex[index]
                    babeli=rowindextoatomindex[i]

                    bond=mol.GetBond(babelindex,babeli)
                    if bond!=None:
                        neighbs.append(i)
            indextoneighborindexes[index]=neighbs
    
    neighborindexes.append(indextoneighborindexes)
    return neighborindexes,indextoneighborindexes

def GrabMatrixIndexes(mat):
    allindexes=[]
    for i in range(len(mat)):
        allindexes.append(i)
    return allindexes

def GenerateAllPairs(allindexes,indextomolecule):
    combs=combinations(allindexes,2)
    allpairs=[]
    for comb in combs:
        temp=[comb[0],comb[1]]
        firstmolecule=indextomolecule[comb[0]]
        secondmolecule=indextomolecule[comb[1]]
        if temp not in allpairs:
            allpairs.append(temp)
    return allpairs

def GenerateReferenceDistances(allpairs,indextomolecule,moleculetoOBmol,rowindextoatomindex,indextoreferencecoordinate,indextovdwradius,newpairwiselocs,pairwisedistanceratio,stericcutoff,inputcoordinates,moleculetoatomicpairtodistance,pairwisetol):
    indexpairtoreferencedistance={}
    indexpairtobounds={}
    for pair in allpairs:
        firstmoleculeatomidx=pair[0]
        secondmoleculeatomidx=pair[1]
        firstmoleculename=indextomolecule[firstmoleculeatomidx]
        secondmoleculename=indextomolecule[secondmoleculeatomidx]
        atomidxpair=[]
        for idx in pair:
            moleculename=indextomolecule[idx]
            mol=moleculetoOBmol[moleculename]
            atomidx=rowindextoatomindex[idx]
            atomidxpair.append(atomidx)
            referenceatom=mol.GetAtom(atomidx)
            atomicnum=referenceatom.GetAtomicNum()
            referencecoordinate=np.array([referenceatom.GetX(),referenceatom.GetY(),referenceatom.GetZ()])
            randomtrans=np.array([np.random.uniform(.2,5),np.random.uniform(.2,5),np.random.uniform(.2,5)])
            indextoreferencecoordinate[idx]=referencecoordinate+randomtrans
        if firstmoleculename!=secondmoleculename:
            firstatomvdw=indextovdwradius[firstmoleculeatomidx]
            secondatomvdw=indextovdwradius[secondmoleculeatomidx]
            if list(pair) in newpairwiselocs or list(pair)[::-1] in newpairwiselocs:
                referencedistance=pairwisedistanceratio*(firstatomvdw+secondatomvdw) 
                indexpairtobounds[tuple(pair)]=[referencedistance-pairwisetol,referencedistance+pairwisetol]

            else:
                referencedistance=stericcutoff
                indexpairtobounds[tuple(pair)]=[stericcutoff,100]
        else:
            atomicpairtodistance=moleculetoatomicpairtodistance[firstmoleculename]
            babelpair=tuple([atomidxpair[0],atomidxpair[1]])
            referencedistance=atomicpairtodistance[babelpair]
            indexpairtobounds[tuple(pair)]=[referencedistance,referencedistance]
        indexpairtoreferencedistance[tuple(pair)]=referencedistance

    return indexpairtoreferencedistance,indexpairtobounds,indextoreferencecoordinate

def Generate3DStructures(matarray,totalindextomoleculearray,totalmoleculetoindexlistarray,totalrowindextoatomindexarray,totalindextovdwradiusarray,totalindextoelectronegativityarray,moleculenametodics,outputfileformat,pairwisedistanceratio,stericcutoff,outputfilepath,basename,dontmove=False,inputcoordinates=None,dontmovesterics=False):
    moleculetoatomindextoatomicsymbol=moleculenametodics['atomindextoatomicsymbol']
    moleculetoOBmol=moleculenametodics['OBmol']
    atomicsymboltovdwradius=moleculenametodics['atomicsymboltovdwradius']
    moleculetoatomicpairtodistance=moleculenametodics['atomicpairtodistance']
    moleculetonumberofatoms=moleculenametodics['numberofatoms']
    moleculetoatomindextosymclass=moleculenametodics['atomindextosymclass']
    labeltodonoracceptorlabel=moleculenametodics['labeltodonoracceptorlabel']
    filenamearray=[]
    targetangle=23
    pairwisetol=.01
    copiesmoleculelist=list(moleculetoOBmol.keys())
    originalcoordsarray=[]
    neighborindexesarray=[]
    narray=[]
    timearray=[]
    prevmatsize=1
    for matidx in tqdm(range(len(matarray)),'Generating 3D Structures'):
        mat=matarray[matidx]
        indextomolecule=totalindextomoleculearray[matidx]
        rowindextoatomindex=totalrowindextoatomindexarray[matidx]
        moleculetoindexlist=totalmoleculetoindexlistarray[matidx]
        indextovdwradius=totalindextovdwradiusarray[matidx]
        indextoelectronegativity=totalindextoelectronegativityarray[matidx]
        indextoreferencecoordinate={}
        moleculelist=[]
        for moleculename in indextomolecule.values():
            if moleculename not in moleculelist:
                moleculelist.append(moleculename)
        matsize=len(moleculelist)
        if matsize!=prevmatsize:
            starttime=time.time()
          

        pairwiselocs,newpairwiselocs=PairwiseLocations(mat)
        allindexes=GrabMatrixIndexes(mat)
        neighborindexes,indextoneighborindexes=GrabNeighborIndexes(allindexes,indextomolecule,moleculetoindexlist,moleculetoOBmol,rowindextoatomindex)
        neighborindexesarray.append(neighborindexes) 
        allpairs=GenerateAllPairs(allindexes,indextomolecule)
        indexpairtoreferencedistance,indexpairtobounds,indextoreferencecoordinate=GenerateReferenceDistances(allpairs,indextomolecule,moleculetoOBmol,rowindextoatomindex,indextoreferencecoordinate,indextovdwradius,newpairwiselocs,pairwisedistanceratio,stericcutoff,inputcoordinates,moleculetoatomicpairtodistance,pairwisetol)
    
        indextooriginalcoordinate=indextoreferencecoordinate.copy()
        originalcoordsarray.append(indextooriginalcoordinate)


        indicestoreferenceangle,indicestobackandforthindices=GenerateReferenceAngles(pairwiselocs,indextomolecule,indextoneighborindexes,targetangle)
        indexpairtoreferencedistance,newpairwiselocs,indexpairtobounds=ConvertAngleRestraintToDistanceRestraint(indexpairtoreferencedistance,indicestoreferenceangle,newpairwiselocs,indexpairtobounds,pairwisetol,indextoreferencecoordinate,indicestobackandforthindices) 

        coordinatesguess=GenerateCoordinateGuesses(indextoreferencecoordinate)
        
        def PairwiseCostFunction(x):
            func=0
            for indexpair,bounds in indexpairtobounds.items():
                firstindex=indexpair[0]
                secondindex=indexpair[1]
                startfirstindex=3*firstindex
                startsecondindex=3*secondindex     
                firstcoordinate=np.array([x[startfirstindex],x[startfirstindex+1],x[startfirstindex+2]])
                secondcoordinate=np.array([x[startsecondindex],x[startsecondindex+1],x[startsecondindex+2]])       
                distance=np.linalg.norm(firstcoordinate-secondcoordinate)
                referencedistance=indexpairtoreferencedistance[indexpair]
                difference=np.abs(distance-referencedistance)
                lowerbound=bounds[0]
                upperbound=bounds[1]
                if distance<lowerbound or distance>upperbound:
                    func+=difference**2


            return func


        sol = minimize(PairwiseCostFunction, coordinatesguess, method='SLSQP',options={'disp':False, 'maxiter': 1000, 'ftol': 1e-6})
        coords=sol.x

        indextoreferencecoordinate=UpdateCoordinates(coords,indextoreferencecoordinate)


        filenamearray=GenerateStructureFiles(indextoreferencecoordinate,moleculetoOBmol,indextomolecule,rowindextoatomindex,matidx,outputfileformat,filenamearray,moleculetoatomindextoatomicsymbol,moleculetoindexlist)

        if matsize!=prevmatsize:
            end=time.time()
            exec=end-starttime
            timearray.append(exec)
            narray.append(matsize)
            prevmatsize=matsize


    return filenamearray,neighborindexesarray,originalcoordsarray,narray,timearray



def GenerateCoordinateGuesses(indextoreferencecoordinate):
    coordinatesguess=[]
    for i in range(len(indextoreferencecoordinate.keys())):
        coordinate=indextoreferencecoordinate[i]
        x,y,z=coordinate[:]
        coordinatesguess.append(x)
        coordinatesguess.append(y)
        coordinatesguess.append(z)
    return coordinatesguess


def GenerateStructureFiles(indextoreferencecoordinate,moleculetoOBmol,indextomolecule,rowindextoatomindex,matidx,outputfileformat,filenamearray,moleculetoatomindextoatomicsymbol,moleculetoindexlist):
    etab = openbabel.OBElementTable() 
    newmol=openbabel.OBMol()
    mlist=[]
    mnamelist=[]
    for index,coord in indextoreferencecoordinate.items():
        molecule=indextomolecule[index]
        atomindextoatomicsymbol=moleculetoatomindextoatomicsymbol[molecule]
        mol=moleculetoOBmol[molecule]
        mlist.append(mol)
        mnamelist.append(molecule)
        atomindex=rowindextoatomindex[index]
        molatom=mol.GetAtom(atomindex)
        atomicsymb=atomindextoatomicsymbol[atomindex-1]
        atomicnum=etab.GetAtomicNum(atomicsymb)
        molatom.SetVector(coord[0],coord[1],coord[2])
        molatom.SetAtomicNum(atomicnum)
        newmol.AddAtom(molatom)
        newmolatom=newmol.GetAtom(atomindex)
    molecules=list(moleculetoindexlist.keys())
    molprefixes=InnerMoleculeNameArray(molecules)
    moleculetomolprefix=dict(zip(molecules,molprefixes))
    for midx in range(len(mlist)):
        m=mlist[midx]
        mname=mnamelist[midx]
        mnamesplit=mname.split('.')
        innermname=mnamesplit[0]
        innermnamesplit=innermname.split('_')
        prefix=innermnamesplit[0]
        samemols=KeysForValues(prefix,moleculetomolprefix)
        newrowindextoatomindex=GrabMappedIndicesFromSameMoleculeTypes(rowindextoatomindex,moleculetoindexlist,samemols)
        bonditer=openbabel.OBMolBondIter(m)
        for bond in bonditer:
           bgnatom=bond.GetBeginAtom()
           bgnatomindex=bgnatom.GetIdx()
           endatom=bond.GetEndAtom()
           endatomindex=endatom.GetIdx()
           bondorder=bond.GetBO()
           bgnkeys=KeysForValues(bgnatomindex,newrowindextoatomindex)
           endkeys=KeysForValues(endatomindex,newrowindextoatomindex)
           zipped=zip(bgnkeys,endkeys) 
           for ls in zipped:
               firstidx=ls[0]+1
               secondidx=ls[1]+1
               newmol.AddBond(firstidx,secondidx,bondorder)
   
    filename='Structure'+'_'+str(matidx)+'.'+outputfileformat 
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat(outputfileformat)
    obConversion.WriteFile(newmol,filename)
    filenamearray.append(filename)
    return filenamearray


def GrabMappedIndicesFromSameMoleculeTypes(rowindextoatomindex,moleculetoindexlist,samemols):
    newrowindextoatomindex={}
    for mol in samemols:
        indexlist=moleculetoindexlist[mol]
        for index in indexlist:
            atomindex=rowindextoatomindex[index]
            newrowindextoatomindex[index]=atomindex
    return newrowindextoatomindex


def KeysForValues(atomindex,rowindextoatomindex):
    keys=[]
    for rowindex,atmindex in rowindextoatomindex.items():
        if atmindex==atomindex:
            keys.append(rowindex)
    return keys

def UpdateCoordinates(coords,indextoreferencecoordinate):
    for i in range(len(indextoreferencecoordinate.keys())):
        startindex=3*i
        coordinate=np.array([coords[startindex],coords[startindex+1],coords[startindex+2]])
        indextoreferencecoordinate[i]=coordinate
    return indextoreferencecoordinate



def ConvertAngleRestraintToDistanceRestraint(indexpairtoreferencedistance,indicestoreferenceangle,pairwiselocs,indexpairtobounds,pairwisetol,indextoreferencecoordinate,indicestobackandforthindices):
    for indices,targetangle in indicestoreferenceangle.items(): 
        donorneighbindex=indices[1]
        acceptorindex=indices[2]
        donorindex=indices[0]
        inputpair=tuple([donorneighbindex,donorindex])
        inputdistance=GrabPairwiseDistance(inputpair,indexpairtoreferencedistance)
        targetpair=tuple([donorindex,acceptorindex])
        targetdistance=GrabPairwiseDistance(targetpair,indexpairtoreferencedistance)
        acceptorcoordinate=indextoreferencecoordinate[acceptorindex]
        donorcoordinate=indextoreferencecoordinate[donorindex]
        donorneighbcoordinate=indextoreferencecoordinate[donorneighbindex]
        acceptordonorvector=donorcoordinate-acceptorcoordinate
        acceptordonorneighbvector=donorneighbcoordinate-acceptorcoordinate
        acceptordonorvectornormed=acceptordonorvector/np.linalg.norm(acceptordonorvector)
        acceptordonorneighbvectornormed=acceptordonorneighbvector/np.linalg.norm(acceptordonorneighbvector)
        currentangle=np.arccos(np.dot(acceptordonorneighbvectornormed,acceptordonorvectornormed))
        angle=180-currentangle-targetangle
        angledist=LawOfCosines(inputdistance,targetdistance,angle)    
        anglepair=tuple([donorneighbindex,acceptorindex])
        indexpairtoreferencedistance[anglepair]=angledist
        pairwiselocs.append(list(anglepair))
        indexpairtobounds[anglepair]=[angledist-pairwisetol,angledist+pairwisetol]
        if indices in indicestobackandforthindices.keys():
            newangle=targetangle+currentangle
            backandforthindices=indicestobackandforthindices[indices]
            acceptorneighbindex=backandforthindices[0]
            newtargetpair=tuple([donorneighbindex,acceptorneighbindex])
            newtargetdistance=GrabPairwiseDistance(newtargetpair,indexpairtoreferencedistance)
            newangledist=LawOfCosines(inputdistance,newtargetdistance,newangle)    
            newanglepair=tuple([donorindex,acceptorneighbindex])
            indexpairtoreferencedistance[newanglepair]=newangledist
            pairwiselocs.append(list(newanglepair))
            indexpairtobounds[newanglepair]=[newangledist-pairwisetol,newangledist+pairwisetol]

   

    return indexpairtoreferencedistance,pairwiselocs,indexpairtobounds

def GrabPairwiseDistance(pair,indexpairtoreferencedistance):
    if pair in indexpairtoreferencedistance.keys():
        distance=indexpairtoreferencedistance[pair]
    elif pair[::-1] in indexpairtoreferencedistance.keys():
        distance=indexpairtoreferencedistance[pair[::-1]]
    return distance


def LawOfCosines(a,b,angleC):
    return np.sqrt(a**2+b**2-2*a*b*np.cos(np.radians(angleC)))




           

def DrawAll3DImages(filenamearray,matarray,molsperrow,graphlabelsarray,neighborindexes,inputbasename,showaxes=False):
    from pymol import cmd,preset
    from PIL import Image
    from pymol.vfont import plain
    from pymol.cgo import CYLINDER,cyl_text
    molsPerImage=molsperrow**2
    imagesize=400

    filenamechunks=ChunksList(Chunks(filenamearray,molsPerImage))
    matarraychunks=ChunksList(Chunks(matarray,molsPerImage))
    labelchunks=ChunksList(Chunks(graphlabelsarray,molsPerImage))
    neighborindexchunks=ChunksList(Chunks(neighborindexes,molsPerImage))
    prevmatslen=len(matarraychunks[0])
    for i in tqdm(range(len(neighborindexchunks)),'Plotting 3D Images'):
        filenamesublist=filenamechunks[i]
        matsublist=matarraychunks[i]
        labelssublist=labelchunks[i]
        neighborindexessublist=neighborindexchunks[i]
        imagenames=[]
        for j in range(len(neighborindexessublist)):
            filename=filenamesublist[j]
            mat=matsublist[j]
            labeldic=labelssublist[j]
            indextoneighborindexes=neighborindexessublist[j][0]
            ls=range(len(filenamesublist))
            chunks=ChunksList(Chunks(ls,molsperrow))
            indextorow={}
            for rowidx in range(len(chunks)):
                row=chunks[rowidx]
                for j in row:
                    indextorow[j]=rowidx

            pairwiselocs=[]
            locs=np.transpose(np.where(mat==1))
            for loc in locs:
                k=loc[0]
                l=loc[1]
                if mat[l,k]==1: # then bond and dont append
                    continue
                else:
                    pairwiselocs.append(loc)

            fileprefix=filename.split('.')[0]
            imagename=fileprefix+'_3D.'+'png'
            imagenames.append(imagename)
            cmd.delete('all')
            cmd.load(filename)
            preset.ball_and_stick(selection='all', mode=1)
            cmd.bg_color("white")
            cmd.set('label_size',26) 
            cmd.set('depth_cue',0)
            cmd.set('ray_trace_fog',0) 
            
            cmd.ray(imagesize,imagesize)
            atomnum=cmd.count_atoms()
            for index in range(atomnum):
                label=labeldic[str(index)]
                labelsplit=label.split('-')
                newlabel='-'.join(labelsplit[1:])
                lab=str(index+1)
                cmd.select("index "+lab)
                cmd.label('sele',lab)
            for pair in pairwiselocs:
                firstidx=pair[0]+1
                secondidx=pair[1]+1
                firstlab=str(firstidx)
                secondlab=str(secondidx)
                cmd.distance(firstlab+'-'+secondlab,'index '+firstlab,'index '+secondlab)
                neighbindexes=indextoneighborindexes[str(pair[0])]
                if len(neighbindexes)==1: # then this is like hydrogen case and can make angle
                    thirdlab=str(int(neighbindexes[0])+1)
                    cmd.angle(name=firstlab+'-'+thirdlab+'-'+secondlab,selection1='index '+firstlab,selection2='index '+thirdlab,selection3='index '+secondlab)
            obj = [
               CYLINDER, 0., 0., 0., 10., 0., 0., 0.2, 1.0, 1.0, 1.0, 1.0, 0.0, 0.,
               CYLINDER, 0., 0., 0., 0., 10., 0., 0.2, 1.0, 1.0, 1.0, 0., 1.0, 0.,
               CYLINDER, 0., 0., 0., 0., 0., 10., 0.2, 1.0, 1.0, 1.0, 0., 0.0, 1.0,
               ]
            
            
            cyl_text(obj,plain,[-5.,-5.,-1],'Origin',0.20,axes=[[3,0,0],[0,3,0],[0,0,3]])
            cyl_text(obj,plain,[10.,0.,0.],'X',0.20,axes=[[3,0,0],[0,3,0],[0,0,3]])
            cyl_text(obj,plain,[0.,10.,0.],'Y',0.20,axes=[[3,0,0],[0,3,0],[0,0,3]])
            cyl_text(obj,plain,[0.,0.,10.],'Z',0.20,axes=[[3,0,0],[0,3,0],[0,0,3]])


            # then we load it into PyMOL
            if showaxes==True:
                cmd.load_cgo(obj,'axes')
            cmd.zoom()
            cmd.png(imagename, imagesize,imagesize)
            cmd.save(fileprefix+'_3D.'+'pse')
        if i>0:
            factor=1
        else:
            factor=0
        firstj=i*prevmatslen+factor
        secondj=firstj+len(matsublist)-factor
        prevmatslen=len(matsublist)

        basename=inputbasename+'_'+str(firstj)+'-'+str(secondj)
        indextoimage={}
        for index in range(len(neighborindexessublist)):
            imagename=imagenames[index]
            image=Image.open(imagename)
            indextoimage[index]=image
        dest = Image.new('RGB', (imagesize*molsperrow,imagesize*molsperrow))
        for j in range(len(neighborindexessublist)):
            row=indextorow[j]
            x=(j-molsperrow*(row))*imagesize
            y=(row)*imagesize
            dest.paste(indextoimage[j],(x,y))
        dest.show()
        dest.save(basename+'.png')

def CalculateSymmetry(pmol, frag_atoms, symmetry_classes):
    """
    Intent: Uses and builds on openbabel's 'GetGIVector' method which,
    "Calculates a set of graph invariant indexes using the graph theoretical distance,
    number of connected heavy atoms, aromatic boolean,
    ring boolean, atomic number, and summation of bond orders connected to the atom", to
    find initial symmetry classes.
    Input: 
        pmol: OBMol object 
        frag_atoms: OBBitVec object containing information about the largest fragment in 'pmol' 
        symmetry_classes: the symmetry_classes array which will be filled in
    Output: 
		nclasses: # of classes
        symmetry_classes: array is filled in
    Referenced By: gen_canonicallabels
    Description:
    1. vectorUnsignedInt object is created, 'vgi'
    2. It is filled in with the call to GetGIVector
    3. The 'symmetry_classes' array is initially filled in to match with 'vgi'
    4. These initial invariant classes do not suit our needs perfectly,
       so the ExtendInvariants method is called to find the more 
       refined classes that we need
    """

    vgi = openbabel.vectorUnsignedInt()
    natoms = pmol.NumAtoms()
    nfragatoms = frag_atoms.CountBits()
    pmol.GetGIVector(vgi)
    iteratom = openbabel.OBMolAtomIter(pmol)
    for atom in iteratom:
        idx = atom.GetIdx()
        if(frag_atoms.BitIsOn(idx)):
            symmetry_classes.append([atom, vgi[idx-1]])
    nclasses = ExtendInvariants(pmol, symmetry_classes,frag_atoms,nfragatoms,natoms)
    return nclasses

def ExtendInvariants(pmol, symmetry_classes,frag_atoms,nfragatoms,natoms):
    """
    Intent: Refine the invariants found by openbabel's GetGIVector
    Input: 
        pmol: OBMol object
        symmetry_classes: symmetry classes array
        frag_atoms: vector containing information about which atoms belong to the largest fragment
        nfragatoms: # of atoms belonging to the largest fragment
        natoms: # of atoms belonging to the molecule
    Output: 
        nclasses1: # of symmetry classes found 
        symmetry_classes: this array is updated
    Referenced By: CalculateSymmetry
    Description:
    
    Description:
    1. Find the # of current classes found by openbabel, nclasses1, 
       and renumber (relabel) the classes to 1, 2, 3, ...
    2. Begin loop
       a. CreateNewClassVector is called which fills in the 'tmp_classes' array with
          a new set of classes by considering bonding information
       b. The number of classes in tmp_classes is found, 'nclasses2'
       c. If there was no change, nclasses1 == nclasses2, break
       d. If the number of classes changed, set nclasses1 to nclasses2, then continue loop
       e. The loop is continued because now that the symmetry classes have changed,
          CreateNewClassVector may find more classes
    3. Return # of classes found
    """
    nclasses1 = CountAndRenumberClasses(symmetry_classes)
    tmp_classes = []
    if(nclasses1 < nfragatoms):
        #stops when number of classes don't change
        for i in range(100):
            CreateNewClassVector(pmol,symmetry_classes, tmp_classes, frag_atoms, natoms)
            nclasses2 = CountAndRenumberClasses(tmp_classes)
            del symmetry_classes[:]
            symmetry_classes.extend(tmp_classes)
            if(nclasses1 == nclasses2):
                break
            nclasses1 = nclasses2
    return nclasses1

def CountAndRenumberClasses(symmetry_classes):
    """
    Intent: Counts the number of symmetry classes and renumbers them to 1, 2, 3, ...
    Input: 
        symmetry_classes: Array of symmetry classes
    Output: 
        count: # of symmetry classes
        symmetry_classes array is updated
    Referenced By: ExtendInvariants
    Description: -
    """
    count = 1
    symmetry_classes = sorted(symmetry_classes, key=lambda sym: sym[1])
    if(len(symmetry_classes) > 0):
        idatom = symmetry_classes[0][1]
        symmetry_classes[0][1] = 1
        for i in range(1,len(symmetry_classes)):
            if(symmetry_classes[i][1] != idatom):
                idatom = symmetry_classes[i][1]
                count = count + 1
                symmetry_classes[i][1] = count
            else:
                symmetry_classes[i][1] = count
    return count

def CreateNewClassVector(pmol,symmetry_classes, tmp_classes, frag_atoms, natoms):
    """
    Intent: Find new symmetry classes if possible
    If two atoms were originally of the same sym class but are bound to atoms of differing
    sym classes, then these two atoms will now belong to two different sym classes
    Input:
        pmol: OBMol object
        symmetry_classes: previous set of symmetry classes
        tmp_classes: tmp array of new symmetry classes
        frag_atoms: atoms in largest fragment
        natoms: number of atoms
    Ouptut:
        tmp_classes is edited
    Referenced By: ExtendInvariants
    Description:
    1. dict idx2index is created which maps atom idx's to an index ranging
       from 0 to # of symmetry classes - 1
    2. For each atom a:
           a. For each bond b that a belongs to:
               i. Find the idx of the other atom, nbratom, in the bond b
               ii. Find and append the symmetry class that nbratom belongs to to vtmp
           b. Using vtmp, create a label for atom a
              i. This label contains information about the symmetry classes of the atoms
              that a is bound to
              ii. This label will be different for two atoms that were originally the same 
              symmetry class but are bound to atoms of differing symmetry classes
    """
    idx2index = dict()
    index = 0
    del tmp_classes[:]
    for s in symmetry_classes:
        idx2index.update({s[0].GetIdx() : index})
        index = index + 1
    for s in symmetry_classes:
        iterbond = openbabel.OBMolBondIter(pmol)
        atom = s[0]
        idatom = s[1]
        nbridx =  0
        vtmp = []
        for b in iterbond:
            #if atom belongs to bond b
            if atom.GetIdx() == b.GetEndAtomIdx() or atom.GetIdx() == b.GetBeginAtomIdx():
                if(atom.GetIdx() == b.GetEndAtomIdx()):
                    nbridx = b.GetBeginAtomIdx()
                elif(atom.GetIdx() == b.GetBeginAtomIdx()):
                    nbridx = b.GetEndAtomIdx()
                if(frag_atoms.BitIsOn(nbridx)):
                    vtmp.append(symmetry_classes[idx2index[nbridx]][1])
        vtmp.sort()
        m = 100
        for v in vtmp:
            idatom = idatom + v * m
            m = 100 * m
        tmp_classes.append([atom,idatom])


def gen_canonicallabels(mol,symclassesused):
    symmetryclass = [ 0 ] * mol.NumAtoms()
    """
    Intent: Find the symmetry class that each atom belongs to
    Input: 
        mol: OBMol object 
    Output: 
        The global variable 'symmetryclass' is altered
    Referenced By: main
    Description:
    1. An empty bit vector is created, 'frag_atoms'
    2. OBMol.FindLargestFragment is called to fill in the 'frag_atoms' bit vector (the
    vector is filled with a 1 or 0 depending on whether the atom is part of the largest
    fragment or not)
    3. 'CalculateSymmetry' method is called to find initial symmetry classes
    4. Terminal atoms of the same element are collapsed to one symmetry class
    5. Possibly renumber the symmetry classes
    """
    # Returns symmetry classes for each atom ID
    frag_atoms = openbabel.OBBitVec()
    symmclasslist = []
    mol.FindLargestFragment(frag_atoms)
    CalculateSymmetry(mol, frag_atoms, symmclasslist)
    for ii in range(len(symmetryclass)):
        symmetryclass[ii] = symmclasslist[ii][1]

    # Collapse terminal atoms of same element to one type
    for a in openbabel.OBMolAtomIter(mol):
        for b in openbabel.OBAtomAtomIter(a):
            if b.GetValence() == 1:
                for c in openbabel.OBAtomAtomIter(a):
                    if ((b is not c) and
                        (c.GetValence() == 1) and
                        (b.GetAtomicNum() == c.GetAtomicNum()) and
                        (symmetryclass[b.GetIdx()-1] !=
                            symmetryclass[c.GetIdx()-1])):
                        symmetryclass[c.GetIdx()-1] = \
                            symmetryclass[b.GetIdx()-1]

    # Renumber symmetry classes
    allcls=list(set(symmetryclass))
    allcls.sort()
    for ii in range(len(symmetryclass)):
        symmetryclass[ii] = allcls.index(symmetryclass[ii]) + 1
    idxtosymclass={}
    for i in range(len(symmetryclass)):
        j=i+1
        symmclass=get_class_number(j,symmetryclass,symclassesused)
        idxtosymclass[j]=symmclass
    zeroindexedidxtosymclass={}
    for idx,symclass in idxtosymclass.items():
        zeroindexedidxtosymclass[idx-1]=symclass
    return zeroindexedidxtosymclass,symmetryclass



def get_class_number(idx,symmetryclass,symclassesused):
    """
    Intent: Given an atom idx, return the atom's class number
    """
    maxidx =  max(symmetryclass)
    if len(symclassesused)==0:
        startidx=1
    else:
        startidx=max(symclassesused)+1

    return startidx + (maxidx - symmetryclass[idx - 1])


def GenerateMolCopy(mol):
    newmol=openbabel.OBMol()
    oldindextonewindex={}
    newidx=1
    for i in range(1,mol.NumAtoms()+1):
        oldatom=mol.GetAtom(i)
        newmol.AddAtom(oldatom)
        newmol.AddAtom(oldatom)
        oldindextonewindex[i]=newidx
        newidx+=1
    bonditer=openbabel.OBMolBondIter(mol)
    for bond in bonditer:
        oendidx = bond.GetEndAtomIdx()
        obgnidx = bond.GetBeginAtomIdx()
        endidx=oldindextonewindex[oendidx]
        bgnidx=oldindextonewindex[obgnidx]
        diditwork=newmol.AddBond(bgnidx,endidx,bond.GetBondOrder())
        if diditwork==False:
            raise ValueError("could not add copied bond from molecule " +str(bgnidx)+','+str(endidx)+" to new molecule ")

    return newmol

def ReadInStructuresAndMatrices(outputfileformat,basename):
    filenamearray=[]
    matarray=[]
    indextofilename={}
    indextomat={}
    files=os.listdir()
    for f in files:
        filesplit=f.split('.')
        ext=filesplit[1]
        if ext=='txt' and basename in f:
            prefix=filesplit[0]
            indexsplit=prefix.split('_')
            index=int(indexsplit[1])
            mat=np.loadtxt(f)
            indextomat[index]=mat

        elif ext==outputfileformat and basename in f:
            prefix=filesplit[0]
            indexsplit=prefix.split('_')
            index=int(indexsplit[1])
            indextofilename[index]=f
    for index in sorted(indextofilename.keys()):
        structfilename=indextofilename[index]
        mat=indextomat[index]
        filenamearray.append(structfilename)
        matarray.append(mat)
    return filenamearray,matarray


def ProjectPairwisePointsOnToPlane(referencepoint,normalvector,pairwiselocs,indextoreferencecoordinate):
    indextoprojectedcoordinate={}
    for pair in pairwiselocs: 
        for index in pair:
            point=indextoreferencecoordinate[index]
            projected_point=ProjectPointOnToPlane(referencepoint,normalvector,point)
            indextoprojectedcoordinate[index]=projected_point             
    return indextoprojectedcoordinate
 
def ProjectPointOnToPlane(referencepoint,normalvector,point):
    v=point-referencepoint
    dist=np.dot(v,normalvector)
    projected_point=point-dist*normalvector
    return projected_point 


def UpdateCoordinatesViaRotation(indextoreferencecoordinate,params):
    euler1,euler2,euler3=params[:]
    rot = Rotation.from_euler("xyz", [euler1, euler2, euler3], degrees=True)
    for index,referencecoordinate in indextoreferencecoordinate.items():
        newreferencecoordinate=rot.apply(referencecoordinate)
        indextoreferencecoordinate[index]=newreferencecoordinate
    return indextoreferencecoordinate

def GeneratePlane(atomindex1,atomindex2,atomindex3,indextoreferencecoordinate):
    coordinate1=np.array(indextoreferencecoordinate[atomindex1])
    coordinate2=np.array(indextoreferencecoordinate[atomindex2])
    coordinate3=np.array(indextoreferencecoordinate[atomindex3])
    secondtofirst=coordinate1-coordinate2
    secondtothird=coordinate3-coordinate2
    normalvector=np.cross(secondtofirst,secondtothird)
    normalvector=normalvector/np.linalg.norm(normalvector)
    referencepoint=coordinate2
    return referencepoint,normalvector 

def AtomIndexToNumberOfPairwiseInteractions(pairwiselocs,allindexes):
    indextonumberofpairwiseinteractions={}
    for index in allindexes:
        indextonumberofpairwiseinteractions[index]=0
       
    for loc in pairwiselocs:
       for index in loc:
           indextonumberofpairwiseinteractions[index]+=1
    return indextonumberofpairwiseinteractions

def ChoosePlaneAtomIndices(indextonumberofpairwiseinteractions,indextoreferencecoordinate):
    sortedindextonumberofpairwiseinteractions={k: v for k, v in sorted(indextonumberofpairwiseinteractions.items(), key=lambda item: item[1],reverse=True)}
    sortedindices=list(sortedindextonumberofpairwiseinteractions.keys())
    atomindex1=sortedindices[0]
    atomindex2=sortedindices[1]
    atomindex3=sortedindices[2]
    return atomindex1,atomindex2,atomindex3


def SumProjectedPairwiseDistances(indextoprojectedcoordinate,pairwiselocs):
    Sum=0
    for pair in pairwiselocs:
        firstindex=pair[0]
        secondindex=pair[1]
        firstcoord=indextoprojectedcoordinate[firstindex]
        secondcoord=indextoprojectedcoordinate[secondindex]
        distance=np.linalg.norm(secondcoord-firstcoord) 
        Sum+=distance
    return Sum





if gen3Dimages==True:
    if not os.path.exists(os.getcwd()+'/'+outputfilepath):
        os.mkdir(os.getcwd()+'/'+outputfilepath)
        os.chdir(os.getcwd()+'/'+outputfilepath)
    else:
        os.chdir(outputfilepath)

    filenamearray,matarray=ReadInStructuresAndMatrices(outputfileformat,'Structure')
    graphlabelsarray=json.load(open("GraphLabels.txt"))
    neighborindexes=json.load(open('NeighborIndexes.txt'))
    DrawAll3DImages(filenamearray,matarray,molsperrow,graphlabelsarray,neighborindexes,'Structures_3D')

else:
    import time
    import matplotlib.pyplot as plt
    import networkx as nx
    from itertools import product
    from itertools import permutations
    import openbabel
    from itertools import combinations
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles
    from rdkit.Geometry import Point3D
    from scipy.optimize import minimize,basinhopping
    from scipy.optimize import NonlinearConstraint
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    import svgutils.transform as sg
    from cairosvg import svg2png
    import random
    import multiprocessing
    from joblib import Parallel, delayed
    from numpy import dot
    from numpy.linalg import eig
    from numpy import diag
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances
    from rdkit.Chem import AllChem
    from scipy.spatial.transform import Rotation
    moleculenametodics=GenerateDictionaries(molecules,donormaxedgesout)
    if not os.path.exists(os.getcwd()+'/'+outputfilepath):
        os.mkdir(os.getcwd()+'/'+outputfilepath)
        os.chdir(os.getcwd()+'/'+outputfilepath)
    else:
        os.chdir(outputfilepath)


    PlotScalingAndUniqueGraphsAndStructuresAllSizes(maxclustersize,moleculenametodics,molsperrow,outputfileformat,pairwisedistanceratio,outputfilepath,stericcutoff,maxgraphinvorder,molecules)
