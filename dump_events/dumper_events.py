# Selection Applied 
# MC_offlineTrigger && (! MC_isLatVetoHit)
# MC_offlineTrigger = isPlaneHit[0][3] and isPlaneHit[1][3] and (TEdep[0]>0.2 or TEdep[1]>0.2 or TEdep[2]>0.2 or TEdep[3]>0.2 or TEdep[4]>0.2 or TEdep[5]>0.2)
# MC_isLatVetoHit = VEdep[0] > 0.2 or VEdep[1] > 0.2 or VEdep[2] > 0.2 or VEdep[3] > 0.2

# python3 dumper_events.py /storage/gpfs_data/limadou/carfora/MC2L2_production/MC2L2_p_10-1000_40x40_cos2theta_flat_optics_total.root protons.pkl List_ev_protons.npz
# python3 dumper_events.py /storage/gpfs_data/limadou/carfora/MC2L2_production/MC2L2_e_1-200_40x40_cos2theta_flat_optics_total.root electrons.pkl List_ev_electrons.npz

from ROOT import gROOT, TFile, TTree, TMath, TCut, TChain
import numpy as np
import pickle
import sys
import pandas as pd

def isBotHit(VEdep):
	#botHit = VEdep[4] > 0.2
	botHit = np.array(list(VEdep))>0.2
	return botHit*1.

def trig_bool(TEdep):
	trig_bool = np.array(list(TEdep))>0.2
	return trig_bool*1.

def offlinesel_MC(isPlaneHit,TEdep,VEdep):
	#MC_offlineTrigger = isPlaneHit[0][3] and isPlaneHit[1][3] and isPlaneHit[2][3] and isPlaneHit[3][3] and (TEdep[0]>0.2 or TEdep[1]>0.2 or TEdep[2]>0.2 or TEdep[3]>0.2 or TEdep[4]>0.2 or TEdep[5]>0.2)
	MC_offlineTrigger = isPlaneHit[0][3] and isPlaneHit[1][3] and (TEdep[0]>0.2 or TEdep[1]>0.2 or TEdep[2]>0.2 or TEdep[3]>0.2 or TEdep[4]>0.2 or TEdep[5]>0.2)
	#MC_VetoHit  = VEdep[0] > 0.2 or VEdep[1] > 0.2 or VEdep[2] > 0.2 or VEdep[3] > 0.2 or VEdep[4] > 0.2
	off_trig = MC_offlineTrigger# and (not(MC_VetoHit))
	return off_trig

def SumLysoEne(lysovec):
	LysoEne = lysovec[0]+lysovec[1]+lysovec[2]+lysovec[3]+lysovec[4]+lysovec[5]+lysovec[6]+lysovec[7]+lysovec[8]
	return LysoEne

def Load_event_list(file):
	event_id = np.load(file)
	print(event_id['ev_list'])
	return np.asarray(event_id['ev_list'])

def EventDump(TRIGBAR, PMT_HG, LYSO, TOTALEdep, energy, PID):#,theta, phi, x, y, z, TOWEREdep, LysoEdep):
	#event = dict({"TRIGBAR":TRIGBAR,"PMT_HG":PMT_HG,"LYSO":LYSO,"energy":energy,"theta":theta,"phi":phi,"x":x,"y":y,"z":z,"TOWEREdep":TOWEREdep,"LysoEdep":LysoEdep})
	event = TRIGBAR+PMT_HG+LYSO+list(TOTALEdep)+list(energy)+list(PID)
	return event

if __name__ == '__main__':
	gROOT.Reset()
	
	input_ROOT_file1 = sys.argv[1]
	input_ROOT_file2 = sys.argv[2]
	outFILEName = sys.argv[3]

	root_file1 = TFile(input_ROOT_file1)
	root_file2 = TFile(input_ROOT_file2)
	t_MCtr = root_file1.Get("L2MCtr")
	t_SIGtr = root_file1.Get("L2")



	input_ev_id_file,event_id = [],[]
	event_form = []

	isPlaneHit = np.ndarray( (16,5), dtype='bool', buffer=t_SIGtr.isPlaneHit)

	sel_eve = 0
	nentries = t_MCtr.GetEntries()
	for i in np.arange(0,nentries):
		if (i % 1000) == 0:
			print(str(i)+'/'+str(t_MCtr.GetEntries()))
		t_MCtr.GetEntry(i)
		t_SIGtr.GetEntry(i)
		if not(offlinesel_MC(isPlaneHit,t_MCtr.TEdep,t_MCtr.VEdep)):
			continue
		sel_eve = sel_eve + 1
		#energy, theta, phi, x, y, z, TOWEREdep, LysoEdep, PMT_HG, LYSO, TRIG
		planeSigHG = list(t_SIGtr.planeSigHG)
		lysoCrystalSig = list(t_SIGtr.lysoCrystalSig)
		trigHit = list(trig_bool(t_MCtr.TEdep))
		event_form.append(EventDump(trigHit,planeSigHG,lysoCrystalSig,[t_MCtr.TOTALEdep],[t_MCtr.energy],[0]))#,t_MCtr.theta,t_MCtr.phi,t_MCtr.gen[0],t_MCtr.gen[1],t_MCtr.gen[2],t_MCtr.TRIGEdep,t_MCtr.TOWEREdep,SumLysoEne(t_MCtr.LEdep)))
	print(sel_eve)

	t_MCtr = root_file2.Get("L2MCtr")
	t_SIGtr = root_file2.Get("L2")
	
	isPlaneHit = np.ndarray( (16,5), dtype='bool', buffer=t_SIGtr.isPlaneHit)

	sel_eve = 0
	nentries = t_MCtr.GetEntries()
	for i in np.arange(0,nentries):
		if (i % 1000) == 0:
			print(str(i)+'/'+str(t_MCtr.GetEntries()))
		t_MCtr.GetEntry(i)
		t_SIGtr.GetEntry(i)
		if not(offlinesel_MC(isPlaneHit,t_MCtr.TEdep,t_MCtr.VEdep)):
			continue
		sel_eve = sel_eve + 1
		#energy, theta, phi, x, y, z, TOWEREdep, LysoEdep, PMT_HG, LYSO, TRIG
		planeSigHG = list(t_SIGtr.planeSigHG)
		lysoCrystalSig = list(t_SIGtr.lysoCrystalSig)
		trigHit = list(trig_bool(t_MCtr.TEdep))
		event_form.append(EventDump(trigHit,planeSigHG,lysoCrystalSig,[t_MCtr.TOTALEdep],[t_MCtr.energy],[1]))#,t_MCtr.theta,t_MCtr.phi,t_MCtr.gen[0],t_MCtr.gen[1],t_MCtr.gen[2],t_MCtr.TRIGEdep,t_MCtr.TOWEREdep,SumLysoEne(t_MCtr.LEdep)))
	print(sel_eve)

	df = pd.DataFrame(event_form)
	# shuffle the dataframe
	df = df.sample(frac=1).reset_index(drop=True)
	df_train = df[:int((0.8)*len(event_form))]
	df_validation = df[int((0.8)*len(event_form)):int((0.9)*len(event_form))]
	df_test = df[int((0.9)*len(event_form)):]

	df_train.to_csv(outFILEName+'_train.csv', index=False)
	df_validation.to_csv(outFILEName+'_validation.csv', index=False)
	df_test.to_csv(outFILEName+'_test.csv', index=False)



