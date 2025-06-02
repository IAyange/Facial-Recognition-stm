import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../utils/preferences_service.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  TextEditingController nameController = TextEditingController();
  TextEditingController addictionController = TextEditingController();
  TextEditingController recoveryStageController = TextEditingController();
  TextEditingController goalsController = TextEditingController();
  String selectedGender = "Male";
  bool hasSupportSystem = true;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    loadProfileData();
  }

  Future<void> loadProfileData() async {
    setState(() => _isLoading = true);
    nameController.text = await PreferencesService.getData('name') ?? "";
    addictionController.text =
        await PreferencesService.getData('addiction_type') ?? "";
    recoveryStageController.text =
        await PreferencesService.getData('recovery_stage') ?? "";
    goalsController.text = await PreferencesService.getData('goals') ?? "";
    selectedGender = await PreferencesService.getData('gender') ?? "Male";
    hasSupportSystem =
        (await PreferencesService.getData('support_system')) == "Yes";
    setState(() => _isLoading = false);
  }

  Future<void> saveProfileData() async {
    setState(() => _isLoading = true);
    try {
      await PreferencesService.saveData('name', nameController.text);
      await PreferencesService.saveData(
          'addiction_type', addictionController.text);
      await PreferencesService.saveData(
          'recovery_stage', recoveryStageController.text);
      await PreferencesService.saveData('goals', goalsController.text);
      await PreferencesService.saveData('gender', selectedGender);
      await PreferencesService.saveData(
          'support_system', hasSupportSystem ? "Yes" : "No");

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text("Profile updated successfully!"),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          backgroundColor: Theme.of(context).primaryColor,
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text("Error saving profile: ${e.toString()}"),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _signOut() async {
    try {
      await _auth.signOut();
      if (!mounted) return;
      Navigator.pushNamedAndRemoveUntil(context, '/login', (route) => false);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text("Error signing out: ${e.toString()}"),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: const Text("Profile Settings",
            style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: _signOut,
            tooltip: 'Logout',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  // Profile Picture Section
                  _buildProfileHeader(),
                  const SizedBox(height: 30),

                  // Personal Information Section
                  _buildSectionHeader("Personal Information"),
                  const SizedBox(height: 15),
                  _buildProfileField("Full Name", nameController, Icons.person),
                  _buildProfileField("Addiction Type", addictionController,
                      Icons.health_and_safety),

                  // Recovery Information Section
                  _buildSectionHeader("Recovery Information"),
                  const SizedBox(height: 15),
                  _buildProfileField("Stage of Recovery",
                      recoveryStageController, Icons.timeline),
                  _buildProfileField("Goals", goalsController, Icons.flag),

                  // Additional Information Section
                  _buildSectionHeader("Additional Information"),
                  const SizedBox(height: 15),
                  _buildGenderDropdown(),
                  _buildSupportSystemToggle(),

                  // Save Button
                  const SizedBox(height: 30),
                  _buildSaveButton(),
                ],
              ),
            ),
    );
  }

  Widget _buildProfileHeader() {
    return Center(
      child: Column(
        children: [
          Stack(
            alignment: Alignment.bottomRight,
            children: [
              Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: Theme.of(context).primaryColor,
                    width: 3,
                  ),
                ),
                child: ClipOval(
                  child: Image.asset(
                    'assets/default_profile.png',
                    fit: BoxFit.cover,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: const BoxDecoration(
                  color: Colors.white,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.edit,
                  color: Theme.of(context).primaryColor,
                  size: 20,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ValueListenableBuilder(
            valueListenable: nameController,
            builder: (context, value, child) {
              return Text(
                nameController.text.isEmpty ? "Your Name" : nameController.text,
                style: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              );
            },
          ),
          Text(
            addictionController.text.isEmpty
                ? "Addiction type"
                : addictionController.text,
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey[600],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Text(
        title,
        style: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: Theme.of(context).primaryColor,
        ),
      ),
    );
  }

  Widget _buildProfileField(
      String label, TextEditingController controller, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 15),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, color: Colors.grey[600]),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey[300]!),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey[300]!),
          ),
          filled: true,
          fillColor: Colors.white,
        ),
      ),
    );
  }

  Widget _buildGenderDropdown() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 15),
      child: DropdownButtonFormField<String>(
        value: selectedGender,
        decoration: InputDecoration(
          labelText: "Gender",
          prefixIcon: Icon(Icons.transgender, color: Colors.grey[600]),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey[300]!),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey[300]!),
          ),
          filled: true,
          fillColor: Colors.white,
        ),
        items: ["Male", "Female", "Other", "Prefer not to say"]
            .map((gender) => DropdownMenuItem(
                  value: gender,
                  child: Text(gender),
                ))
            .toList(),
        onChanged: (value) {
          setState(() {
            selectedGender = value!;
          });
        },
      ),
    );
  }

  Widget _buildSupportSystemToggle() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey[300]!),
      ),
      child: ListTile(
        leading: Icon(Icons.group, color: Colors.grey[600]),
        title: const Text("Have a support system?"),
        trailing: Switch.adaptive(
          value: hasSupportSystem,
          onChanged: (value) {
            setState(() {
              hasSupportSystem = value;
            });
          },
          activeColor: Theme.of(context).primaryColor,
        ),
      ),
    );
  }

  Widget _buildSaveButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: saveProfileData,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          backgroundColor: Theme.of(context).primaryColor,
          elevation: 0,
        ),
        child: const Text(
          "SAVE CHANGES",
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
      ),
    );
  }
}
