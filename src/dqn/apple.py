import jax

# This should verify Metal is detected
print(jax.devices()) 

# This triggers the code that previously failed
key = jax.random.PRNGKey(0)
print("Key created successfully:", key)