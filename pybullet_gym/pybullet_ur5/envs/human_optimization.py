import numpy as np


l0 = 0.4
ls = 0.14
lua=lla=0.22
def right_shoulder(theta, inp_sr, inp_er, disp=False):
	assert len(theta)

	c = [np.cos(t) for t in theta]
	s = [np.sin(t) for t in theta]

	p_sr = np.asarray([-ls * c[0] * c[1],
	l0 - ls * s[1],
	ls * c[1] * s[0]])

	p_er = np.asarray([
		lua * c[2] * c[0] * c[1] * s[4] - lua * s[2] * s[4] * s[0] - ls * c[0] * c[1] + lua * c[3] * c[4] * c[0] * s[
			1] + lua * c[2] * c[4] * s[3] * s[0] + lua * c[4] * c[0] * c[1] * s[2] * s[3],
	l0 - ls * s[1] - lua * c[3] * c[4] * c[1] + lua * c[2] * s[4] * s[1] + lua * c[4] * s[2] * s[3] * s[1],
	ls * c[1] * s[0] - lua * c[0] * s[2] * s[4] + lua * c[2] * c[4] * c[0] * s[3] - lua * c[2] * c[1] * s[4] * s[
		0] - lua * c[3] * c[4] * s[0] * s[1] - lua * c[4] * c[1] * s[2] * s[3] * s[0]])

	A = p_sr-inp_sr
	B = p_er-inp_er

	objective = np.matmul(A, A.T)+np.matmul(B, B.T)
	if disp:
		print("sl {}, el {}".format(p_sr, p_er))
		print("-------------right objective: {}------------------".format(objective))


	return objective

def right_elbow(theta_unknown, theta_shoulder, inp_wl, disp=False):
	c = [np.cos(t) for t in theta_shoulder]
	s = [np.sin(t) for t in theta_shoulder]

	c_u = [np.cos(t1) for t1 in theta_unknown]
	s_u = [np.sin(t1) for t1 in theta_unknown]

	p_wr = np.asarray([
		lua * c[2] * c[0] * c[1] * s[4] - lua * s[2] * s[4] * s[0] - ls * c[0] * c[1] + lua * c[3] * c[4] * c[0] * s[
			1] + lla * c[2] * c[3] * s_u[1] * s[0] + lua * c[2] * c[4] * s[3] * s[0] - lla * c[0] * s_u[1] * s[3] * s[
			1] + lla * c[3] * c[0] * c[1] * s_u[1] * s[2] + lua * c[4] * c[0] * c[1] * s[2] * s[3] - lla * c_u[0] * c_u[
			1] * s[2] * s[4] * s[0] + lla * c_u[1] * c[4] * s_u[0] * s[2] * s[0] + lla * c_u[1] * c[3] * c[0] * s_u[0] *
		s[4] * s[1] + lla * c_u[1] * c[2] * s_u[0] * s[3] * s[4] * s[0] + lla * c_u[0] * c_u[1] * c[2] * c[0] * c[1] *
		s[4] - lla * c_u[1] * c[2] * c[4] * c[0] * c[1] * s_u[0] + lla * c_u[0] * c_u[1] * c[3] * c[4] * c[0] * s[
			1] + lla * c_u[0] * c_u[1] * c[2] * c[4] * s[3] * s[0] + lla * c_u[0] * c_u[1] * c[4] * c[0] * c[1] * s[2] *
		s[3] + lla * c_u[1] * c[0] * c[1] * s_u[0] * s[2] * s[3] * s[4],
	l0 - ls * s[1] - lua * c[3] * c[4] * c[1] + lla * c[1] * s_u[1] * s[3] + lua * c[2] * s[4] * s[1] + lla * c[3] * \
	s_u[1] * s[2] * s[1] + lua * c[4] * s[2] * s[3] * s[1] - lla * c_u[0] * c_u[1] * c[3] * c[4] * c[1] + lla * c_u[0] * \
	c_u[1] * c[2] * s[4] * s[1] - lla * c_u[1] * c[2] * c[4] * s_u[0] * s[1] - lla * c_u[1] * c[3] * c[1] * s_u[0] * s[
		4] + lla * c_u[0] * c_u[1] * c[4] * s[2] * s[3] * s[1] + lla * c_u[1] * s_u[0] * s[2] * s[3] * s[4] * s[1],
	ls * c[1] * s[0] - lua * c[0] * s[2] * s[4] + lla * c[2] * c[3] * c[0] * s_u[1] + lua * c[2] * c[4] * c[0] * s[
		3] - lua * c[2] * c[1] * s[4] * s[0] - lua * c[3] * c[4] * s[0] * s[1] + lla * s_u[1] * s[3] * s[0] * s[
		1] - lla * c_u[0] * c_u[1] * c[0] * s[2] * s[4] + lla * c_u[1] * c[4] * c[0] * s_u[0] * s[2] - lla * c[3] * c[
		1] * s_u[1] * s[2] * s[0] - lua * c[4] * c[1] * s[2] * s[3] * s[0] + lla * c_u[1] * c[2] * c[0] * s_u[0] * s[
		3] * s[4] - lla * c_u[1] * c[3] * s_u[0] * s[4] * s[0] * s[1] + lla * c_u[0] * c_u[1] * c[2] * c[4] * c[0] * s[
		3] - lla * c_u[0] * c_u[1] * c[2] * c[1] * s[4] * s[0] + lla * c_u[1] * c[2] * c[4] * c[1] * s_u[0] * s[
		0] - lla * c_u[0] * c_u[1] * c[3] * c[4] * s[0] * s[1] - lla * c_u[1] * c[1] * s_u[0] * s[2] * s[3] * s[4] * s[
		0] - lla * c_u[0] * c_u[1] * c[4] * c[1] * s[2] * s[3] * s[0]])



	C = p_wr-inp_wl

	objective = np.matmul(C,C.T)
	if disp:
		print("inp_wl {}, wl {}".format(inp_wl, p_wr))
		print("-------------right objective: {}------------------".format(objective))
	return objective



def left_shoulder(theta, inp_sl, inp_el, disp=False):
	'''
	# theta_ssy = theta[0];
	# theta_ssz = theta[1];
	# theta_slx = theta[2];
	# theta_sly = theta[3];
	# theta_slz = theta[4];
	# theta_elx = theta[5];
	# theta_elz = theta[6];
	'''

	assert len(theta)
	# lua = np.linalg.norm(inp_el - inp_sl)
	# lla = np.linalg.norm(inp_wl - inp_el)

	c = [np.cos(t) for t in theta]
	s = [np.sin(t) for t in theta]

	p_sl = np.asarray([ls * c[0] * c[1],
		l0 + ls * s[1],
		-ls * c[1] * s[0]])

	p_el = np.asarray([
		ls * c[0] * c[1] + lua * s[2] * s[4] * s[0] + lua * c[2] * c[0] * c[1] * s[4] + lua * c[3] * c[4] * c[0] * s[
			1] - lua * c[2] * c[4] * s[3] * s[0] + lua * c[4] * c[0] * c[1] * s[2] * s[3],
	l0 + ls * s[1] - lua * c[3] * c[4] * c[1] + lua * c[2] * s[4] * s[1] + lua * c[4] * s[2] * s[3] * s[1],
	lua * c[0] * s[2] * s[4] - ls * c[1] * s[0] - lua * c[2] * c[4] * c[0] * s[3] - lua * c[2] * c[1] * s[4] * s[
		0] - lua * c[3] * c[4] * s[0] * s[1] - lua * c[4] * c[1] * s[2] * s[3] * s[0]])


	A = p_sl-inp_sl
	B = p_el-inp_el


	objective = np.matmul(A, A.T)+np.matmul(B, B.T)
	if disp:
		print("sl {}, el {}".format(p_sl, p_el))
		print("-------------left objective: {}------------------".format(objective))

	return objective

def left_elbow(theta_unknown, theta_shoulder, inp_wl, disp=False):


	c = [np.cos(t) for t in theta_shoulder]
	s = [np.sin(t) for t in theta_shoulder]

	c_u = [np.cos(t1) for t1 in theta_unknown]
	s_u = [np.sin(t1) for t1 in theta_unknown]

	p_wl = np.asarray([
		ls * c[0] * c[1] + lua * s[2] * s[4] * s[0] + lua * c[2] * c[0] * c[1] * s[4] + lua * c[3] * c[4] * c[0] * s[
			1] - lla * c[2] * c[3] * s_u[1] * s[0] - lua * c[2] * c[4] * s[3] * s[0] - lla * c[0] * s_u[1] * s[3] * s[
			1] + lla * c[3] * c[0] * c[1] * s_u[1] * s[2] + lua * c[4] * c[0] * c[1] * s[2] * s[3] + lla * c_u[0] * c_u[
			1] * s[2] * s[4] * s[0] + lla * c_u[1] * c[4] * s_u[0] * s[2] * s[0] - lla * c_u[1] * c[3] * c[0] * s_u[0] *
		s[4] * s[1] + lla * c_u[1] * c[2] * s_u[0] * s[3] * s[4] * s[0] + lla * c_u[0] * c_u[1] * c[2] * c[0] * c[1] *
		s[4] + lla * c_u[1] * c[2] * c[4] * c[0] * c[1] * s_u[0] + lla * c_u[0] * c_u[1] * c[3] * c[4] * c[0] * s[
			1] - lla * c_u[0] * c_u[1] * c[2] * c[4] * s[3] * s[0] + lla * c_u[0] * c_u[1] * c[4] * c[0] * c[1] * s[2] *
		s[3] - lla * c_u[1] * c[0] * c[1] * s_u[0] * s[2] * s[3] * s[4],
	l0 + ls * s[1] - lua * c[3] * c[4] * c[1] + lla * c[1] * s_u[1] * s[3] + lua * c[2] * s[4] * s[1] + lla * c[3] * \
	s_u[1] * s[2] * s[1] + lua * c[4] * s[2] * s[3] * s[1] - lla * c_u[0] * c_u[1] * c[3] * c[4] * c[1] + lla * c_u[0] * \
	c_u[1] * c[2] * s[4] * s[1] + lla * c_u[1] * c[2] * c[4] * s_u[0] * s[1] + lla * c_u[1] * c[3] * c[1] * s_u[0] * s[
		4] + lla * c_u[0] * c_u[1] * c[4] * s[2] * s[3] * s[1] - lla * c_u[1] * s_u[0] * s[2] * s[3] * s[4] * s[1],
	lua * c[0] * s[2] * s[4] - ls * c[1] * s[0] - lla * c[2] * c[3] * c[0] * s_u[1] - lua * c[2] * c[4] * c[0] * s[
		3] - lua * c[2] * c[1] * s[4] * s[0] - lua * c[3] * c[4] * s[0] * s[1] + lla * s_u[1] * s[3] * s[0] * s[
		1] + lla * c_u[0] * c_u[1] * c[0] * s[2] * s[4] + lla * c_u[1] * c[4] * c[0] * s_u[0] * s[2] - lla * c[3] * c[
		1] * s_u[1] * s[2] * s[0] - lua * c[4] * c[1] * s[2] * s[3] * s[0] + lla * c_u[1] * c[2] * c[0] * s_u[0] * s[
		3] * s[4] + lla * c_u[1] * c[3] * s_u[0] * s[4] * s[0] * s[1] - lla * c_u[0] * c_u[1] * c[2] * c[4] * c[0] * s[
		3] - lla * c_u[0] * c_u[1] * c[2] * c[1] * s[4] * s[0] - lla * c_u[1] * c[2] * c[4] * c[1] * s_u[0] * s[
		0] - lla * c_u[0] * c_u[1] * c[3] * c[4] * s[0] * s[1] + lla * c_u[1] * c[1] * s_u[0] * s[2] * s[3] * s[4] * s[
		0] - lla * c_u[0] * c_u[1] * c[4] * c[1] * s[2] * s[3] * s[0]])


	C = p_wl-inp_wl

	objective = np.matmul(C,C.T)
	if disp:
		print("inp_wl {}, wl {}".format(inp_wl, p_wl))
		print("-------------left objective: {}------------------".format(objective))


	return objective
