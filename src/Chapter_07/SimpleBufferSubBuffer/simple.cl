kernel void square(global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * buffer[id];
}
