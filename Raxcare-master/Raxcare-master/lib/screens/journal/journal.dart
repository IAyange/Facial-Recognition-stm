import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:intl/intl.dart';

class JournalScreen extends StatefulWidget {
  @override
  _JournalScreenState createState() => _JournalScreenState();
}

class _JournalScreenState extends State<JournalScreen> {
  final Journal _journalService = Journal();
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _titleController;
  late TextEditingController _contentController;
  DateTime? _selectedDate;
  String? _editingDocId;
  bool _isEditing = false;

  @override
  void initState() {
    super.initState();
    _titleController = TextEditingController();
    _contentController = TextEditingController();
    _selectedDate = DateTime.now();
  }

  @override
  void dispose() {
    _titleController.dispose();
    _contentController.dispose();
    super.dispose();
  }

  void _showJournalForm({DocumentSnapshot? journal}) async {
    setState(() {
      _isEditing = journal != null;
      if (_isEditing) {
        _titleController.text = journal!['title'];
        _contentController.text = journal['content'];
        _selectedDate = (journal['date'] as Timestamp).toDate();
        _editingDocId = journal.id;
      }
    });

    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        child: Container(
          padding: EdgeInsets.all(20),
          child: Form(
            key: _formKey,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _isEditing ? 'Edit Entry' : 'New Journal Entry',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 20),
                TextFormField(
                  controller: _titleController,
                  decoration: InputDecoration(
                    labelText: 'Title',
                    border: OutlineInputBorder(),
                  ),
                  validator: (value) =>
                      value!.isEmpty ? 'Title is required' : null,
                ),
                SizedBox(height: 15),
                TextFormField(
                  controller: _contentController,
                  maxLines: 5,
                  decoration: InputDecoration(
                    labelText: 'Content',
                    border: OutlineInputBorder(),
                  ),
                  validator: (value) =>
                      value!.isEmpty ? 'Content is required' : null,
                ),
                SizedBox(height: 15),
                ListTile(
                  title: Text(
                    'Date: ${DateFormat('MMM dd, yyyy').format(_selectedDate!)}',
                  ),
                  trailing: Icon(Icons.calendar_today),
                  onTap: () async {
                    final pickedDate = await showDatePicker(
                      context: context,
                      initialDate: _selectedDate!,
                      firstDate: DateTime(2000),
                      lastDate: DateTime.now(),
                    );
                    if (pickedDate != null) {
                      setState(() => _selectedDate = pickedDate);
                    }
                  },
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () async {
                    if (_formKey.currentState!.validate()) {
                      try {
                        if (_isEditing) {
                          await _journalService.updateJournal(
                            _editingDocId!,
                            _titleController.text,
                            _contentController.text,
                            _selectedDate!,
                          );
                        } else {
                          await _journalService.addJournal(
                            _titleController.text,
                            _contentController.text,
                            _selectedDate!,
                          );
                        }
                        Navigator.pop(context);
                        _clearForm();
                      } catch (e) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('Error saving journal: $e'),
                            backgroundColor: Colors.red,
                          ),
                        );
                      }
                    }
                  },
                  child: Text(_isEditing ? 'Update' : 'Save'),
                  style: ElevatedButton.styleFrom(
                    minimumSize: Size(double.infinity, 50),
                  ),
                ),
                const SizedBox(height: 10.0),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _clearForm() {
    _titleController.clear();
    _contentController.clear();
    _selectedDate = DateTime.now();
    _editingDocId = null;
    _isEditing = false;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('My Journal'),
        elevation: 0,
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: _journalService.getJournals(),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }

          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          }

          final journals = snapshot.data!.docs;

          if (journals.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.book, size: 60, color: Colors.grey),
                  SizedBox(height: 20),
                  Text(
                    'No journal entries yet',
                    style: TextStyle(fontSize: 18, color: Colors.grey),
                  ),
                  SizedBox(height: 10),
                  Text(
                    'Tap the + button to add your first entry',
                    style: TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            );
          }

          return ListView.builder(
            padding: EdgeInsets.all(16),
            itemCount: journals.length,
            itemBuilder: (context, index) {
              final journal = journals[index];
              final data = journal.data() as Map<String, dynamic>;
              final date = (data['date'] as Timestamp).toDate();

              return Card(
                margin: EdgeInsets.only(bottom: 16),
                elevation: 2,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                child: InkWell(
                  borderRadius: BorderRadius.circular(12),
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => JournalDetailScreen(
                        title: data['title'],
                        content: data['content'],
                        date: date,
                      ),
                    ),
                  ),
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: Text(
                                data['title'],
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            PopupMenuButton(
                              itemBuilder: (context) => [
                                PopupMenuItem(
                                  child: Text('Edit'),
                                  value: 'edit',
                                ),
                                PopupMenuItem(
                                  child: Text('Delete'),
                                  value: 'delete',
                                ),
                              ],
                              onSelected: (value) async {
                                if (value == 'edit') {
                                  _showJournalForm(journal: journal);
                                } else if (value == 'delete') {
                                  await _journalService
                                      .deleteJournal(journal.id);
                                }
                              },
                            ),
                          ],
                        ),
                        SizedBox(height: 8),
                        Text(
                          data['content'],
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(color: Colors.grey[600]),
                        ),
                        SizedBox(height: 8),
                        Text(
                          DateFormat('MMM dd, yyyy').format(date),
                          style: TextStyle(
                            color: Colors.grey,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            },
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showJournalForm(),
        child: Icon(Icons.add),
        backgroundColor: Theme.of(context).primaryColor,
      ),
    );
  }
}

class Journal {
  final CollectionReference journals =
      FirebaseFirestore.instance.collection('journals');

  Future<void> addJournal(String title, String content, DateTime date) async {
    try {
      await journals.add({
        'title': title,
        'content': content,
        'date': date,
        'userId': FirebaseAuth.instance.currentUser!.uid,
        'createdAt': FieldValue.serverTimestamp(),
      });
    } catch (e) {
      throw Exception('Failed to add journal: $e');
    }
  }

  Stream<QuerySnapshot> getJournals() {
    return journals
        .where('userId', isEqualTo: FirebaseAuth.instance.currentUser!.uid)
        .orderBy('date', descending: true)
        .snapshots();
  }

  Future<void> updateJournal(
      String docId, String title, String content, DateTime date) async {
    try {
      await journals.doc(docId).update({
        'title': title,
        'content': content,
        'date': date,
        'updatedAt': FieldValue.serverTimestamp(),
      });
    } catch (e) {
      throw Exception('Failed to update journal: $e');
    }
  }

  Future<void> deleteJournal(String docId) async {
    try {
      await journals.doc(docId).delete();
    } catch (e) {
      throw Exception('Failed to delete journal: $e');
    }
  }
}

class JournalDetailScreen extends StatelessWidget {
  final String title;
  final String content;
  final DateTime date;

  JournalDetailScreen({
    required this.title,
    required this.content,
    required this.date,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Journal Entry'),
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              DateFormat('MMMM dd, yyyy').format(date),
              style: TextStyle(
                color: Colors.grey,
                fontSize: 14,
              ),
            ),
            SizedBox(height: 8),
            Text(
              title,
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 24),
            Text(
              content,
              style: TextStyle(
                fontSize: 16,
                height: 1.6,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
