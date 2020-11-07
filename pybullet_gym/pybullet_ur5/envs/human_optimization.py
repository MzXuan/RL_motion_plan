import numpy as np


l0 = 0.4
ls = 0.14
# lua=lla=0.22
def right(theta, inp_sr, inp_er, inp_wr, disp=False):
	assert len(theta)

	lua = np.linalg.norm(inp_er-inp_sr)
	lla = np.linalg.norm(inp_wr-inp_er)

	c = [np.cos(t) for t in theta]
	s = [np.sin(t) for t in theta]

	p_sr =np.asarray([-ls * c[0] * c[1], l0 - ls * s[1], ls * c[1] * s[0]])

	p_er =np.asarray([ls * c[2] * c[4] - ls + lua * c[2] * c[1] * s[4] + ls * c[2] * s[4] * s[1] - ls * s[2] * s[3] * s[4] - ls * c[2] * c[
		4] * c[0] * c[1] + lua * c[2] * c[4] * c[0] * s[1] + ls * c[3] * c[1] * s[2] * s[0] + lua * c[4] * c[1] * s[2] * s[
		3] + ls * c[4] * s[2] * s[3] * s[1] - lua * c[3] * s[2] * s[0] * s[1] + ls * c[0] * c[1] * s[2] * s[3] * s[
		4] - lua * c[0] * s[2] * s[3] * s[4] * s[1],
	l0 + ls * c[3] * s[4] - lua * c[3] * c[4] * c[1] - ls * c[3] * c[4] * s[1] + ls * c[1] * s[3] * s[0] - lua * s[3] * s[
		0] * s[1] - ls * c[3] * c[0] * c[1] * s[4] + lua * c[3] * c[0] * s[4] * s[1],
	ls * c[2] * c[3] * c[1] * s[0] - ls * c[2] * s[3] * s[4] - lua * c[1] * s[2] * s[4] - ls * s[2] * s[4] * s[1] - ls * c[
		4] * s[2] + ls * c[4] * c[0] * c[1] * s[2] + lua * c[2] * c[4] * c[1] * s[3] + ls * c[2] * c[4] * s[3] * s[
		1] - lua * c[2] * c[3] * s[0] * s[1] - lua * c[4] * c[0] * s[2] * s[1] + ls * c[2] * c[0] * c[1] * s[3] * s[
		4] - lua * c[2] * c[0] * s[3] * s[4] * s[1]])

	p_wr =np.asarray([lua * c[6] * s[5] - ls + ls * c[5] * c[2] * c[4] + lla * c[5] * c[2] * c[1] * s[4] + lua * c[5] * c[2] * c[1] * s[
		4] + ls * c[6] * c[3] * s[5] * s[4] + ls * c[5] * c[2] * s[4] * s[1] - ls * c[4] * s[5] * s[6] * s[2] - ls * c[5] * \
	s[2] * s[3] * s[4] - ls * c[5] * c[2] * c[4] * c[0] * c[1] - lla * c[6] * c[3] * c[4] * c[1] * s[5] + lla * c[5] * c[
		2] * c[4] * c[0] * s[1] - lua * c[6] * c[3] * c[4] * c[1] * s[5] + lua * c[5] * c[2] * c[4] * c[0] * s[1] + lla * c[
		5] * c[4] * c[1] * s[2] * s[3] - ls * c[6] * c[3] * c[4] * s[5] * s[1] + ls * c[5] * c[3] * c[1] * s[2] * s[
		0] + lua * c[5] * c[4] * c[1] * s[2] * s[3] - lla * c[5] * c[3] * s[2] * s[0] * s[1] + ls * c[6] * c[1] * s[5] * s[
		3] * s[0] + ls * c[5] * c[4] * s[2] * s[3] * s[1] - lua * c[5] * c[3] * s[2] * s[0] * s[1] - lla * c[1] * s[5] * s[
		6] * s[2] * s[4] - lla * c[6] * s[5] * s[3] * s[0] * s[1] - ls * c[2] * s[5] * s[6] * s[3] * s[4] - lua * c[1] * s[
		5] * s[6] * s[2] * s[4] - lua * c[6] * s[5] * s[3] * s[0] * s[1] - ls * s[5] * s[6] * s[2] * s[4] * s[1] + lla * c[
		2] * c[4] * c[1] * s[5] * s[6] * s[3] + lla * c[6] * c[3] * c[0] * s[5] * s[4] * s[1] + ls * c[2] * c[3] * c[1] * s[
		5] * s[6] * s[0] + ls * c[4] * c[0] * c[1] * s[5] * s[6] * s[2] + ls * c[5] * c[0] * c[1] * s[2] * s[3] * s[
		4] + lua * c[2] * c[4] * c[1] * s[5] * s[6] * s[3] + lua * c[6] * c[3] * c[0] * s[5] * s[4] * s[1] - lla * c[2] * c[
		3] * s[5] * s[6] * s[0] * s[1] - lla * c[4] * c[0] * s[5] * s[6] * s[2] * s[1] - lla * c[5] * c[0] * s[2] * s[3] * \
	s[4] * s[1] + ls * c[2] * c[4] * s[5] * s[6] * s[3] * s[1] - lua * c[2] * c[3] * s[5] * s[6] * s[0] * s[1] - lua * c[
		4] * c[0] * s[5] * s[6] * s[2] * s[1] - lua * c[5] * c[0] * s[2] * s[3] * s[4] * s[1] - ls * c[6] * c[3] * c[0] * c[
		1] * s[5] * s[4] - lla * c[2] * c[0] * s[5] * s[6] * s[3] * s[4] * s[1] - lua * c[2] * c[0] * s[5] * s[6] * s[3] * \
	s[4] * s[1] + ls * c[2] * c[0] * c[1] * s[5] * s[6] * s[3] * s[4],
	l0 - lua + lua * c[5] * c[6] - ls * c[2] * c[4] * s[5] + ls * c[5] * c[6] * c[3] * s[4] - lla * c[2] * c[1] * s[5] * s[
		4] - ls * c[5] * c[4] * s[6] * s[2] - lua * c[2] * c[1] * s[5] * s[4] - ls * c[2] * s[5] * s[4] * s[1] + ls * s[5] * \
	s[2] * s[3] * s[4] - lla * c[5] * c[6] * c[3] * c[4] * c[1] - lua * c[5] * c[6] * c[3] * c[4] * c[1] - ls * c[5] * c[
		6] * c[3] * c[4] * s[1] + ls * c[2] * c[4] * c[0] * c[1] * s[5] - lla * c[2] * c[4] * c[0] * s[5] * s[1] + ls * c[
		5] * c[6] * c[1] * s[3] * s[0] - lua * c[2] * c[4] * c[0] * s[5] * s[1] - lla * c[5] * c[1] * s[6] * s[2] * s[
		4] - lla * c[5] * c[6] * s[3] * s[0] * s[1] - lla * c[4] * c[1] * s[5] * s[2] * s[3] - ls * c[5] * c[2] * s[6] * s[
		3] * s[4] - ls * c[3] * c[1] * s[5] * s[2] * s[0] - lua * c[5] * c[1] * s[6] * s[2] * s[4] - lua * c[5] * c[6] * s[
		3] * s[0] * s[1] - lua * c[4] * c[1] * s[5] * s[2] * s[3] + lla * c[3] * s[5] * s[2] * s[0] * s[1] - ls * c[5] * s[
		6] * s[2] * s[4] * s[1] - ls * c[4] * s[5] * s[2] * s[3] * s[1] + lua * c[3] * s[5] * s[2] * s[0] * s[1] - lla * c[
		5] * c[2] * c[3] * s[6] * s[0] * s[1] - lla * c[5] * c[4] * c[0] * s[6] * s[2] * s[1] + ls * c[5] * c[2] * c[4] * s[
		6] * s[3] * s[1] - lua * c[5] * c[2] * c[3] * s[6] * s[0] * s[1] - lua * c[5] * c[4] * c[0] * s[6] * s[2] * s[
		1] - ls * c[0] * c[1] * s[5] * s[2] * s[3] * s[4] + lla * c[0] * s[5] * s[2] * s[3] * s[4] * s[1] + lua * c[0] * s[
		5] * s[2] * s[3] * s[4] * s[1] - ls * c[5] * c[6] * c[3] * c[0] * c[1] * s[4] + lla * c[5] * c[2] * c[4] * c[1] * s[
		6] * s[3] + lla * c[5] * c[6] * c[3] * c[0] * s[4] * s[1] + ls * c[5] * c[2] * c[3] * c[1] * s[6] * s[0] + ls * c[
		5] * c[4] * c[0] * c[1] * s[6] * s[2] + lua * c[5] * c[2] * c[4] * c[1] * s[6] * s[3] + lua * c[5] * c[6] * c[3] * \
	c[0] * s[4] * s[1] + ls * c[5] * c[2] * c[0] * c[1] * s[6] * s[3] * s[4] - lla * c[5] * c[2] * c[0] * s[6] * s[3] * s[
		4] * s[1] - lua * c[5] * c[2] * c[0] * s[6] * s[3] * s[4] * s[1],
	(s[6] * (s[3] * s[0] * s[1] + c[3] * c[4] * c[1] - c[3] * c[0] * s[4] * s[1]) - c[6] * (
				c[1] * (s[2] * s[4] - c[2] * c[4] * s[3]) + c[0] * s[1] * (c[4] * s[2] + c[2] * s[3] * s[4]) + c[2] * c[3] *
				s[0] * s[1])) * (lla - l0 + lua) - c[6] * (
				s[2] * (ls * (c[4] - 1) + l0 * s[4]) + ls * s[2] - c[2] * s[3] * (l0 * (c[4] - 1) - ls * s[4]) + l0 * (
					c[1] - 1) * (s[2] * s[4] - c[2] * c[4] * s[3]) - l0 * c[2] * s[3] + l0 * c[0] * s[1] * (
							c[4] * s[2] + c[2] * s[3] * s[4]) + l0 * c[2] * c[3] * s[0] * s[1]) + ls * (
				s[6] * (c[3] * c[4] * s[1] - c[1] * s[3] * s[0] + c[3] * c[0] * c[1] * s[4]) + c[6] * (
					c[0] * c[1] * (c[4] * s[2] + c[2] * s[3] * s[4]) - s[1] * (s[2] * s[4] - c[2] * c[4] * s[3]) + c[2] * c[
				3] * c[1] * s[0])) + s[6] * (
				l0 * (c[3] - 1) + c[3] * (l0 * (c[4] - 1) - ls * s[4]) + l0 * s[3] * s[0] * s[1] + l0 * c[3] * c[4] * (
					c[1] - 1) - l0 * c[3] * c[0] * s[4] * s[1]) + s[6] * (l0 - lua)])

	A = p_sr-inp_sr
	B = p_er-inp_er
	C = p_wr-inp_wr


	objective = np.matmul(A, A.T)+np.matmul(B, B.T)+5*np.matmul(C,C.T)
	if disp:
		print("sl {}, el {}, wl {}".format(p_sr, p_er, p_wr))
		print("-------------right objective: {}------------------".format(objective))


	return objective


def left(theta, inp_sl, inp_el, inp_wl, disp=False):
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
	lua = np.linalg.norm(inp_el - inp_sl)
	lla = np.linalg.norm(inp_wl - inp_el)

	c = [np.cos(t) for t in theta]
	s = [np.sin(t) for t in theta]

	p_sl =np.asarray([ls * c[0] * c[1], l0 + ls * s[1],-ls * c[1] * s[0]])


	p_el =np.asarray([
		ls - ls * c[2] * c[4] + lua * c[2] * c[1] * s[4] - ls * c[2] * s[4] * s[1] + ls * s[2] * s[3] * s[4] + ls * c[
			2] * c[4] * c[0] * c[1] + lua * c[2] * c[4] * c[0] * s[1] + ls * c[3] * c[1] * s[2] * s[0] + lua * c[4] * c[
			1] * s[2] * s[3] - ls * c[4] * s[2] * s[3] * s[1] + lua * c[3] * s[2] * s[0] * s[1] - ls * c[0] * c[1] * s[
			2] * s[3] * s[4] - lua * c[0] * s[2] * s[3] * s[4] * s[1],
	l0 - ls * c[3] * s[4] - lua * c[3] * c[4] * c[1] + ls * c[3] * c[4] * s[1] + ls * c[1] * s[3] * s[0] + lua * s[3] * \
	s[0] * s[1] + ls * c[3] * c[0] * c[1] * s[4] + lua * c[3] * c[0] * s[4] * s[1],
	lua * c[1] * s[2] * s[4] - ls * c[2] * s[3] * s[4] - ls * c[4] * s[2] - ls * s[2] * s[4] * s[1] - ls * c[2] * c[3] *
	c[1] * s[0] + ls * c[4] * c[0] * c[1] * s[2] - lua * c[2] * c[4] * c[1] * s[3] + ls * c[2] * c[4] * s[3] * s[
		1] - lua * c[2] * c[3] * s[0] * s[1] + lua * c[4] * c[0] * s[2] * s[1] + ls * c[2] * c[0] * c[1] * s[3] * s[
		4] + lua * c[2] * c[0] * s[3] * s[4] * s[1]])


	p_wl = np.asarray([

		ls - lua * c[6] * s[5] - ls * c[5] * c[2] * c[4] + lla * c[5] * c[2] * c[1] * s[4] + lua * c[5] * c[2] * c[1] *
		s[4] + ls * c[6] * c[3] * s[5] * s[4] - ls * c[5] * c[2] * s[4] * s[1] - ls * c[4] * s[5] * s[6] * s[2] + ls *
		c[5] * s[2] * s[3] * s[4] + ls * c[5] * c[2] * c[4] * c[0] * c[1] + lla * c[6] * c[3] * c[4] * c[1] * s[
			5] + lla * c[5] * c[2] * c[4] * c[0] * s[1] + lua * c[6] * c[3] * c[4] * c[1] * s[5] + lua * c[5] * c[2] *
		c[4] * c[0] * s[1] + lla * c[5] * c[4] * c[1] * s[2] * s[3] - ls * c[6] * c[3] * c[4] * s[5] * s[1] + ls * c[
			5] * c[3] * c[1] * s[2] * s[0] + lua * c[5] * c[4] * c[1] * s[2] * s[3] + lla * c[5] * c[3] * s[2] * s[0] *
		s[1] - ls * c[6] * c[1] * s[5] * s[3] * s[0] - ls * c[5] * c[4] * s[2] * s[3] * s[1] + lua * c[5] * c[3] * s[
			2] * s[0] * s[1] + lla * c[1] * s[5] * s[6] * s[2] * s[4] - lla * c[6] * s[5] * s[3] * s[0] * s[1] - ls * c[
			2] * s[5] * s[6] * s[3] * s[4] + lua * c[1] * s[5] * s[6] * s[2] * s[4] - lua * c[6] * s[5] * s[3] * s[0] *
		s[1] - ls * s[5] * s[6] * s[2] * s[4] * s[1] - lla * c[2] * c[4] * c[1] * s[5] * s[6] * s[3] - lla * c[6] * c[
			3] * c[0] * s[5] * s[4] * s[1] - ls * c[2] * c[3] * c[1] * s[5] * s[6] * s[0] + ls * c[4] * c[0] * c[1] * s[
			5] * s[6] * s[2] - ls * c[5] * c[0] * c[1] * s[2] * s[3] * s[4] - lua * c[2] * c[4] * c[1] * s[5] * s[6] *
		s[3] - lua * c[6] * c[3] * c[0] * s[5] * s[4] * s[1] - lla * c[2] * c[3] * s[5] * s[6] * s[0] * s[1] + lla * c[
			4] * c[0] * s[5] * s[6] * s[2] * s[1] - lla * c[5] * c[0] * s[2] * s[3] * s[4] * s[1] + ls * c[2] * c[4] *
		s[5] * s[6] * s[3] * s[1] - lua * c[2] * c[3] * s[5] * s[6] * s[0] * s[1] + lua * c[4] * c[0] * s[5] * s[6] * s[
			2] * s[1] - lua * c[5] * c[0] * s[2] * s[3] * s[4] * s[1] - ls * c[6] * c[3] * c[0] * c[1] * s[5] * s[
			4] + lla * c[2] * c[0] * s[5] * s[6] * s[3] * s[4] * s[1] + lua * c[2] * c[0] * s[5] * s[6] * s[3] * s[4] *
		s[1] + ls * c[2] * c[0] * c[1] * s[5] * s[6] * s[3] * s[4],
	l0 - lua + lua * c[5] * c[6] - ls * c[2] * c[4] * s[5] - ls * c[5] * c[6] * c[3] * s[4] + lla * c[2] * c[1] * s[5] * \
	s[4] + ls * c[5] * c[4] * s[6] * s[2] + lua * c[2] * c[1] * s[5] * s[4] - ls * c[2] * s[5] * s[4] * s[1] + ls * s[
		5] * s[2] * s[3] * s[4] - lla * c[5] * c[6] * c[3] * c[4] * c[1] - lua * c[5] * c[6] * c[3] * c[4] * c[1] + ls * \
	c[5] * c[6] * c[3] * c[4] * s[1] + ls * c[2] * c[4] * c[0] * c[1] * s[5] + lla * c[2] * c[4] * c[0] * s[5] * s[
		1] + ls * c[5] * c[6] * c[1] * s[3] * s[0] + lua * c[2] * c[4] * c[0] * s[5] * s[1] - lla * c[5] * c[1] * s[6] * \
	s[2] * s[4] + lla * c[5] * c[6] * s[3] * s[0] * s[1] + lla * c[4] * c[1] * s[5] * s[2] * s[3] + ls * c[5] * c[2] * \
	s[6] * s[3] * s[4] + ls * c[3] * c[1] * s[5] * s[2] * s[0] - lua * c[5] * c[1] * s[6] * s[2] * s[4] + lua * c[5] * \
	c[6] * s[3] * s[0] * s[1] + lua * c[4] * c[1] * s[5] * s[2] * s[3] + lla * c[3] * s[5] * s[2] * s[0] * s[1] + ls * \
	c[5] * s[6] * s[2] * s[4] * s[1] - ls * c[4] * s[5] * s[2] * s[3] * s[1] + lua * c[3] * s[5] * s[2] * s[0] * s[
		1] + lla * c[5] * c[2] * c[3] * s[6] * s[0] * s[1] - lla * c[5] * c[4] * c[0] * s[6] * s[2] * s[1] - ls * c[5] * \
	c[2] * c[4] * s[6] * s[3] * s[1] + lua * c[5] * c[2] * c[3] * s[6] * s[0] * s[1] - lua * c[5] * c[4] * c[0] * s[6] * \
	s[2] * s[1] - ls * c[0] * c[1] * s[5] * s[2] * s[3] * s[4] - lla * c[0] * s[5] * s[2] * s[3] * s[4] * s[1] - lua * \
	c[0] * s[5] * s[2] * s[3] * s[4] * s[1] + ls * c[5] * c[6] * c[3] * c[0] * c[1] * s[4] + lla * c[5] * c[2] * c[4] * \
	c[1] * s[6] * s[3] + lla * c[5] * c[6] * c[3] * c[0] * s[4] * s[1] + ls * c[5] * c[2] * c[3] * c[1] * s[6] * s[
		0] - ls * c[5] * c[4] * c[0] * c[1] * s[6] * s[2] + lua * c[5] * c[2] * c[4] * c[1] * s[6] * s[3] + lua * c[5] * \
	c[6] * c[3] * c[0] * s[4] * s[1] - ls * c[5] * c[2] * c[0] * c[1] * s[6] * s[3] * s[4] - lla * c[5] * c[2] * c[0] * \
	s[6] * s[3] * s[4] * s[1] - lua * c[5] * c[2] * c[0] * s[6] * s[3] * s[4] * s[1],
	(s[6] * (s[3] * s[0] * s[1] - c[3] * c[4] * c[1] + c[3] * c[0] * s[4] * s[1]) + c[6] * (
				c[1] * (s[2] * s[4] - c[2] * c[4] * s[3]) + c[0] * s[1] * (c[4] * s[2] + c[2] * s[3] * s[4]) - c[2] * c[
			3] * s[0] * s[1])) * (lla - l0 + lua) - c[6] * (
				s[2] * (ls * (c[4] - 1) - l0 * s[4]) + ls * s[2] + c[2] * s[3] * (l0 * (c[4] - 1) + ls * s[4]) - l0 * (
					c[1] - 1) * (s[2] * s[4] - c[2] * c[4] * s[3]) + l0 * c[2] * s[3] - l0 * c[0] * s[1] * (
							c[4] * s[2] + c[2] * s[3] * s[4]) + l0 * c[2] * c[3] * s[0] * s[1]) + ls * (
				s[6] * (c[1] * s[3] * s[0] + c[3] * c[4] * s[1] + c[3] * c[0] * c[1] * s[4]) - c[6] * (
					s[1] * (s[2] * s[4] - c[2] * c[4] * s[3]) - c[0] * c[1] * (c[4] * s[2] + c[2] * s[3] * s[4]) + c[
				2] * c[3] * c[1] * s[0])) - s[6] * (
				l0 * (c[3] - 1) + c[3] * (l0 * (c[4] - 1) + ls * s[4]) - l0 * s[3] * s[0] * s[1] + l0 * c[3] * c[4] * (
					c[1] - 1) - l0 * c[3] * c[0] * s[4] * s[1]) - s[6] * (l0 - lua)])

	A = p_sl-inp_sl
	B = p_el-inp_el
	C = p_wl-inp_wl


	objective = np.matmul(A, A.T)+np.matmul(B, B.T)+5*np.matmul(C,C.T)
	if disp:
		print("sl {}, el {}, wl {}".format(p_sl, p_el, p_wl))
		print("-------------left objective: {}------------------".format(objective))


	return objective


	# obj = left_main_obj(theta, inp_sl, inp_el, inp_wl)


# def fk_e(theta):
#
# 	c0 = np.cos(theta[0])
# 	c1 = np.cos(theta[1])
# 	c2 = np.cos(theta[2])
# 	c3 = np.cos(theta[3])
# 	s0 = np.sin(theta[0])
# 	s1 = np.sin(theta[1])
# 	s2 = np.sin(theta[2])
# 	s3 = np.sin(theta[3])
# 	result = np.asarray([(7*c0*s2)/25 + (7*c2*s0*s1)/25 + 9/50,
#                                                        9/20 - (7*c1*c2)/25,
#         (7*s0*s2)/25 - (7*c0*c2*s1)/25])
# 	print("fk _e result; ", result)
#
# def fk_w(theta):
#
# 	c0 = np.cos(theta[0])
# 	c1 = np.cos(theta[1])
# 	c2 = np.cos(theta[2])
# 	c3 = np.cos(theta[3])
# 	s0 = np.sin(theta[0])
# 	s1 = np.sin(theta[1])
# 	s2 = np.sin(theta[2])
# 	s3 = np.sin(theta[3])
# 	result  = np.asarray([(14*c0*s2)/25 + (14*c2*s0*s1)/25 + 9/50,
#  (7*c3)/25 - (14*s3*s0*s2)/25 - (14*c3*c1*c2)/25 + (14*c0*c2*s3*s1)/25 + 17/100,
# (7*s3)/25 + (14*c3*s0*s2)/25 - (14*c1*c2*s3)/25 - (14*c3*c0*c2*s1)/25])
# 	print("fk _w result; ", result)
#
#
# def left(theta, inputP_el, inputP_wl):
#
# 	c0 = np.cos(theta[0])
# 	c1 = np.cos(theta[1])
# 	c2 = np.cos(theta[2])
# 	c3 = np.cos(theta[3])
# 	s0 = np.sin(theta[0])
# 	s1 = np.sin(theta[1])
# 	s2 = np.sin(theta[2])
# 	s3 = np.sin(theta[3])
# 	A = np.asarray([(7*c0*s2)/25 + (7*c2*s0*s1)/25 + 9/50,
#         9/20 - (7*c1*c2)/25,
#         (7*s0*s2)/25 - (7*c0*c2*s1)/25]) -inputP_el
# 	B = np.asarray([(14*c0*s2)/25 + (14*c2*s0*s1)/25 + 9/50,
#  (7*c3)/25 - (14*s3*s0*s2)/25 - (14*c3*c1*c2)/25 + (14*c0*c2*s3*s1)/25 + 17/100,
# (7*s3)/25 + (14*c3*s0*s2)/25 - (14*c1*c2*s3)/25 - (14*c3*c0*c2*s1)/25])-inputP_wl
#
# 	print(np.matmul(A, A.T)+np.matmul(B, B.T))
# 	print("----------")
#
# 	objective = np.matmul(A, A.T)+np.matmul(B, B.T)
#
# 	return objective