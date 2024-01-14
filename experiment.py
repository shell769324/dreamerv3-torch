import envs.crafter as crafter
import itertools

crafter_env = crafter.Crafter(
        "reward", outdir="./"
    )

for name, ind in itertools.chain(crafter_env._env._world._mat_ids.items(), crafter_env._env._sem_view._obj_ids.items()):
    print(name)
    name = str(name)[str(name).find('objects.') + len('objects.'):-2].lower() if 'objects.' in str(
        name) else str(name)
    crafter_env._id_to_item[ind] = name
    print(ind, name)